"""Opt-in transcript replay and signed receipts; no system code is replayed.

Receipts attest score reproducibility from recorded interface observations, not
the truth of those observations. See docs/certification.md for the trust model.
"""

import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

from rsi_bench.core import RSIBenchmark, SystemInterface
from rsi_bench.scoring import UnifiedScorer


EVIDENCE_SCHEMA = "rsi-bench-evidence-v1"
RECEIPT_SCHEMA = "rsi-bench-receipt-v1"
SCOPE = "interface-transcript-replay"
AXES = tuple(UnifiedScorer.DEFAULT_WEIGHTS)
NAMES = {key: name for name, key in UnifiedScorer.AXIS_KEY_MAP.items()}
MAX_CYCLES = 10000
MAX_EVENTS = 1000000


def _json_tree(value):
    """Snapshot JSON values without losing dictionary insertion order."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_json_tree(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _json_tree(v) for k, v in value.items()}
    raise ValueError("Evidence requires finite JSON values and string object keys")


def _encode(value):
    # Order is significant: AGG uses str(dict) to identify goals. Sorting keys
    # would give identical digests to evidence with different replay semantics.
    return json.dumps(_json_tree(value), ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _implementation():
    root = Path(__file__).parent
    sources = [root / "core.py", root / "scoring.py", root / "certification.py"]
    sources += sorted((root / "axes").glob("*.py"))
    return {
        "sources_sha256": hashlib.sha256(b"".join(
            p.relative_to(root).as_posix().encode() + b"\0" +
            p.read_text(encoding="utf-8").encode("utf-8") + b"\0"
            for p in sources)).hexdigest(),
        "python": "{}.{}".format(*sys.version_info[:2]),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }


def _protocol(axes, max_cycles, seed):
    if (type(max_cycles) is not int or not 1 <= max_cycles <= MAX_CYCLES or
            type(seed) is not int or seed < 0):
        raise ValueError("Protocol requires positive bounded cycles and a nonnegative integer seed")
    if (type(axes) is not list or not axes or
            any(type(a) is not str or a not in AXES for a in axes) or
            len(set(axes)) != len(axes)):
        raise ValueError("Protocol requires distinct built-in axes")
    return {"axes": axes, "max_cycles": max_cycles, "seed": seed}


def _claims(results):
    return _json_tree({
        "axes": {key: {"score": results.axis_results[NAMES[key]].score,
                       "metrics": results.axis_results[NAMES[key]].metrics}
                 for key in AXES if NAMES[key] in results.axis_results},
        "composite_score": results.composite_score,
    })


class _Recorder:
    def __init__(self):
        self.events = []
        self.failure = None

    def wrap(self, method, function):
        def call():
            if len(self.events) >= MAX_EVENTS:
                self.failure = "Evidence event limit exceeded"
                raise ValueError(self.failure)
            try:
                value = function()
            except Exception:
                self.events.append({"method": method, "error": True})
                raise
            try:
                snapshot = _json_tree(value)
            except ValueError as exc:
                self.failure = str(exc)
                raise
            self.events.append({"method": method, "value": snapshot})
            # Evaluators may mutate returned collections. Keep the archived
            # observations immutable and avoid aliasing across callback calls.
            return _json_tree(snapshot)
        return call


def record_run(benchmark, max_cycles=50, seed=None, verbose=False):
    """Return (BenchmarkResults, JSON evidence) for an opt-in built-in run.

    Callback values must be JSON snapshots. Ordinary benchmark.run() retains
    its existing behavior. Custom scoring weights are outside the v1 protocol.
    """
    if type(benchmark) is not RSIBenchmark or benchmark.system is None:
        raise ValueError("A registered built-in RSIBenchmark is required")
    if benchmark.scorer.weights != UnifiedScorer.DEFAULT_WEIGHTS:
        raise ValueError("Certification v1 requires default scoring weights")
    protocol = _protocol(list(benchmark._active_axes), max_cycles,
                         benchmark.seed if seed is None else seed)
    original = benchmark.system
    recorder = _Recorder()
    benchmark.system = SystemInterface(
        name=original.name,
        modify_fn=recorder.wrap("modify", original.modify_fn),
        evaluate_fn=recorder.wrap("evaluate", original.evaluate_fn),
        get_state_fn=recorder.wrap("state", original.get_state_fn),
        reset_fn=(recorder.wrap("reset", original.reset_fn)
                  if original.reset_fn is not None else None),
        metadata=original.metadata,
    )
    try:
        results = benchmark.run(max_cycles=max_cycles, seed=protocol["seed"], verbose=verbose)
    finally:
        benchmark.system = original
    if recorder.failure:
        raise ValueError(recorder.failure)
    bundle = _json_tree({
        "schema": EVIDENCE_SCHEMA,
        "implementation": _implementation(),
        "system_name": original.name,
        "protocol": protocol,
        "reset_available": original.reset_fn is not None,
        "events": recorder.events,
        "claims": _claims(results),
    })
    return results, bundle


class _ReplayMismatch(BaseException):
    # SSM deliberately catches callback exceptions. A protocol mismatch must
    # escape that handler instead of being counted as a legitimate crash.
    pass


class _Replay:
    def __init__(self, events):
        self.events = events
        self.position = 0

    def call(self, method):
        if self.position == len(self.events):
            raise _ReplayMismatch("Truncated transcript")
        event = self.events[self.position]
        if event["method"] != method:
            raise _ReplayMismatch("Transcript callback order mismatch")
        self.position += 1
        if event.get("error") is True:
            raise RuntimeError("Recorded callback failure")
        return _json_tree(event["value"])


def _equal(actual, expected):
    if type(actual) in (int, float) and type(expected) in (int, float):
        try:
            return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        except OverflowError:
            return False
    if type(actual) is dict and type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _equal(actual[k], expected[k]) for k in actual)
    return type(actual) is type(expected) and actual == expected


def recompute(bundle):
    """Audit seven fixed claims: six score/metric groups and the composite.

    Missing or mismatching groups count as non-recomputable. Metrics must also
    match, preventing a valid score from laundering a false crash count, etc.
    No original system callbacks or submitted executable code are invoked.
    """
    bundle = _json_tree(bundle)
    required = {"schema", "implementation", "system_name", "protocol",
                "reset_available", "events", "claims"}
    if type(bundle) is not dict or set(bundle) != required or bundle["schema"] != EVIDENCE_SCHEMA:
        raise ValueError("Unsupported evidence schema")
    if bundle["implementation"] != _implementation():
        raise ValueError("Benchmark source or runtime fingerprint mismatch")
    if type(bundle["system_name"]) is not str or not bundle["system_name"]:
        raise ValueError("A system name is required")
    if type(bundle["reset_available"]) is not bool:
        raise ValueError("Invalid reset availability")
    p = bundle["protocol"]
    if type(p) is not dict or set(p) != {"axes", "max_cycles", "seed"}:
        raise ValueError("Unsupported protocol")
    _protocol(p["axes"], p["max_cycles"], p["seed"])
    events = bundle["events"]
    if type(events) is not list or len(events) > MAX_EVENTS:
        raise ValueError("Invalid transcript size")
    for e in events:
        if (type(e) is not dict or e.get("method") not in ("modify", "evaluate", "state", "reset") or
                not (set(e) == {"method", "value"} or
                     (set(e) == {"method", "error"} and e["error"] is True))):
            raise ValueError("Malformed transcript event")
    claimed = bundle["claims"]
    if (type(claimed) is not dict or set(claimed) != {"axes", "composite_score"} or
            type(claimed["axes"]) is not dict or set(claimed["axes"]) - set(AXES)):
        raise ValueError("Unsupported claims")
    replay = _Replay(events)
    bench = RSIBenchmark(axes=p["axes"], seed=p["seed"])
    bench.register_system(
        bundle["system_name"], lambda: replay.call("modify"),
        lambda: replay.call("evaluate"), lambda: replay.call("state"),
        (lambda: replay.call("reset")) if bundle["reset_available"] else None,
    )
    try:
        results = bench.run(max_cycles=p["max_cycles"], seed=p["seed"], verbose=False)
    except (Exception, _ReplayMismatch) as exc:
        raise ValueError("Transcript cannot be replayed: {}".format(exc)) from exc
    if replay.position != len(events):
        raise ValueError("Unconsumed transcript events")
    observed = _claims(results)
    verified = [key for key in AXES if key in observed["axes"] and
                _equal(claimed["axes"].get(key), observed["axes"][key])]
    if _equal(claimed["composite_score"], observed["composite_score"]):
        verified.append("composite")
    integrity = len(verified) / 7.0
    return {
        "scope": SCOPE,
        "system_name": bundle["system_name"],
        "bundle_sha256": _digest(bundle),
        "recomputed": observed,
        "verified_claims": verified,
        "non_recomputable_claims": [k for k in (*AXES, "composite") if k not in verified],
        "integrity_score": integrity,
        "adjusted_score": max(0.0, observed["composite_score"] - (1.0 - integrity)),
        "eligible": len(verified) == 7,
    }


def issue_receipt(bundle, verifier_id, private_key):
    """Recompute before signing. The independent verifier keeps its key private."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("An Ed25519 private key is required")
    if type(verifier_id) is not str or not verifier_id:
        raise ValueError("A verifier ID is required")
    payload = {"schema": RECEIPT_SCHEMA, "verifier_id": verifier_id,
               "audit": recompute(bundle)}
    return {"payload": payload, "signature": private_key.sign(_encode(payload)).hex()}


def verify_receipt(bundle, receipt, trusted_verifiers):
    """Validate against externally configured {verifier_id: raw_public_key_bytes}.

    A key embedded in a submission never establishes trust. Evidence is replayed
    again so a signature over inconsistent scores cannot confer eligibility.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        if set(receipt) != {"payload", "signature"}:
            raise ValueError("Malformed receipt")
        payload = receipt["payload"]
        if (set(payload) != {"schema", "verifier_id", "audit"} or
                payload["schema"] != RECEIPT_SCHEMA):
            raise ValueError("Unsupported receipt schema")
        verifier_id = payload["verifier_id"]
        if verifier_id not in trusted_verifiers:
            raise ValueError("Untrusted verifier")
        public_key = Ed25519PublicKey.from_public_bytes(trusted_verifiers[verifier_id])
        public_key.verify(bytes.fromhex(receipt["signature"]), _encode(payload))
        report = recompute(bundle)
        if _encode(payload["audit"]) != _encode(report):
            raise ValueError("Receipt does not match the evidence audit")
        return report
    except (InvalidSignature, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise ValueError("Invalid receipt") from exc


def rank_submissions(submissions, trusted_verifiers, protocol=None):
    """Return receipt-only ranks and a visible UNVERIFIABLE wall.

    Each submission has an ID, bundle, and optional receipt. Missing/invalid
    receipts, partial runs, and mismatched claims never receive a rank. Every
    ranked run must use the deployment's protocol (default: 50 cycles, seed 42,
    six axes in standard order); evidence cannot choose its own ranking policy.
    """
    ranked, wall = [], []
    seen = set()
    seen_bundles = set()
    protocol = _protocol(list(AXES), 50, 42) if protocol is None else _protocol(
        protocol["axes"], protocol["max_cycles"], protocol["seed"])
    for submission in submissions:
        identity = submission["id"]
        if type(identity) is not str or not identity or identity in seen:
            raise ValueError("Submission IDs must be nonempty and unique")
        seen.add(identity)
        row = {"id": identity, "status": "UNVERIFIABLE"}
        evidence = submission.get("bundle")
        display = evidence if type(evidence) is dict else submission
        if type(display.get("system_name")) is str:
            row["system_name"] = display["system_name"]
        claims = display.get("claims", display)
        reported = claims.get("composite_score") if type(claims) is dict else None
        if type(reported) in (int, float) and 0 <= reported <= 1:
            row["reported_composite_score"] = reported
        try:
            if "bundle" not in submission:
                raise ValueError("Missing callback evidence")
            if "receipt" not in submission:
                raise ValueError("Missing verifier receipt")
            report = verify_receipt(submission["bundle"], submission["receipt"], trusted_verifiers)
            row.update(report)
            if not report["eligible"]:
                raise ValueError("All six axes and composite must be recomputable")
            if submission["bundle"]["protocol"] != protocol:
                raise ValueError("Run does not match the ranking protocol")
            if report["bundle_sha256"] in seen_bundles:
                raise ValueError("Duplicate evidence already ranked")
        except (ValueError, KeyError) as exc:
            row["eligible"] = False
            row["reason"] = str(exc)
            wall.append(row)
            continue
        row["status"] = "VERIFIED_REPLAY"
        seen_bundles.add(report["bundle_sha256"])
        ranked.append(row)
    ranked.sort(key=lambda row: (-row["adjusted_score"], row["id"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return {"ranked": ranked, "unverifiable": wall}
