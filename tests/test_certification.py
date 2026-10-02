"""Receipt trust boundaries and adversarial transcript/claim mutations."""
import copy
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rsi_bench.certification import (
    issue_receipt, rank_submissions, recompute, record_run, verify_receipt,
)
from rsi_bench.core import RSIBenchmark


class ObservedSystem:
    def __init__(self, crash=False):
        self.cycle = 0
        self.crash = crash
        self.modification = None

    def modify(self):
        self.cycle += 1
        # SSM starts after four eight-cycle axes. Exercise both a crash and
        # failed reset, without crashing the preceding unguarded axes.
        if self.crash and self.cycle == 35:
            raise RuntimeError("failed modification")
        self.modification = {
            "level": min(self.cycle // 8, 4),
            "operators": [{"type": "unary", "cross_task": True}],
            "goals": [{"name": "g" * self.cycle, "feasible": True}],
        }
        return self.modification

    def evaluate(self):
        return {"fitness": 1 + self.cycle / 100}

    def state(self):
        return {"new_operators": [{"type": "binary"}],
                "sandbox_violation": self.cycle == 37}

    def reset(self):
        raise RuntimeError("failed reset")


def benchmark(axes=None, crash=False):
    system = ObservedSystem(crash)
    bench = RSIBenchmark(axes=axes)
    bench.register_system("observed", system.modify, system.evaluate,
                          system.state, system.reset)
    return bench


@pytest.fixture
def bundle():
    return record_run(benchmark(), max_cycles=8, seed=0)[1]


@pytest.fixture
def keys():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return private, {"independent": public}


def test_replay_all_axes_after_json_roundtrip(bundle):
    audit = recompute(json.loads(json.dumps(bundle)))
    assert audit["eligible"]
    assert audit["integrity_score"] == 1
    assert audit["adjusted_score"] == bundle["claims"]["composite_score"]
    assert bundle["protocol"]["seed"] == 0


def test_opt_in_matches_ordinary_run_and_restores_interface():
    bench = benchmark()
    original = bench.system
    ordinary = benchmark().run(max_cycles=8, seed=0, verbose=False)
    recorded, bundle = record_run(bench, max_cycles=8, seed=0)
    assert bench.system is original
    assert recorded.composite_score == ordinary.composite_score
    assert recorded.config["seed"] == 0
    assert recompute(bundle)["eligible"]
    # ODR extends the modification's operator list. The archived callback
    # must contain only the original operator, not evaluator mutations.
    assert all(len(e["value"]["operators"]) == 1 for e in bundle["events"]
               if e["method"] == "modify")


def test_crashes_reset_failures_and_violations_survive_replay():
    results, bundle = record_run(benchmark(crash=True), max_cycles=8)
    assert recompute(bundle)["eligible"]
    metrics = results.axis_results["Safety & Stability (SSM)"].metrics
    assert metrics["total_crashes"] == 1
    assert metrics["sandbox_violations"] == 1
    assert {e["method"] for e in bundle["events"] if e.get("error")} == {"modify", "reset"}


@pytest.mark.parametrize("mutation", ["truncate", "append", "reorder", "malformed"])
def test_broken_transcripts_are_rejected(bundle, mutation):
    if mutation == "truncate":
        bundle["events"].pop()
    elif mutation == "append":
        bundle["events"].append({"method": "state", "value": {}})
    elif mutation == "reorder":
        bundle["events"][0], bundle["events"][1] = bundle["events"][1], bundle["events"][0]
    else:
        bundle["events"][0]["error"] = True
    with pytest.raises(ValueError):
        recompute(bundle)


def test_order_mismatch_is_not_laundered_as_safety_crash():
    _, bundle = record_run(benchmark(axes=["ssm"]), max_cycles=8)
    bundle["events"][1]["method"] = "state"
    with pytest.raises(ValueError, match="order mismatch"):
        recompute(bundle)


@pytest.mark.parametrize("mutation", ["score", "metric", "composite", "missing_axis"])
def test_false_and_missing_claims_reduce_integrity(bundle, mutation):
    if mutation == "score":
        bundle["claims"]["axes"]["ssm"]["score"] = 0
    elif mutation == "metric":
        bundle["claims"]["axes"]["ssm"]["metrics"]["total_crashes"] = 10
    elif mutation == "composite":
        bundle["claims"]["composite_score"] = 1
    else:
        del bundle["claims"]["axes"]["ssm"]
    audit = recompute(bundle)
    assert not audit["eligible"]
    assert audit["integrity_score"] == pytest.approx(6 / 7)
    assert audit["adjusted_score"] == pytest.approx(max(
        0, audit["recomputed"]["composite_score"] - 1 / 7))


def test_partial_axes_cannot_enter_certified_ranking(keys):
    _, bundle = record_run(benchmark(axes=["ssm"]), max_cycles=8)
    private, trusted = keys
    receipt = issue_receipt(bundle, "independent", private)
    rows = rank_submissions([{"id": "partial", "bundle": bundle, "receipt": receipt}], trusted)
    assert rows["ranked"] == []
    assert rows["unverifiable"][0]["integrity_score"] == pytest.approx(2 / 7)


def test_trusted_receipt_and_unsigned_wall(bundle, keys):
    private, trusted = keys
    receipt = issue_receipt(bundle, "independent", private)
    assert verify_receipt(bundle, receipt, trusted)["eligible"]
    rows = rank_submissions([
        {"id": "verified", "bundle": bundle, "receipt": receipt},
        {"id": "self-report", "bundle": bundle},
        {"id": "legacy-json", "composite_score": 0.99},
    ], trusted)
    assert rows["ranked"][0]["rank"] == 1
    assert rows["ranked"][0]["status"] == "VERIFIED_REPLAY"
    assert [r["id"] for r in rows["unverifiable"]] == ["self-report", "legacy-json"]
    assert all(r["status"] == "UNVERIFIABLE" and "rank" not in r for r in rows["unverifiable"])
    assert rows["unverifiable"][0]["reported_composite_score"] == bundle["claims"]["composite_score"]
    assert rows["unverifiable"][1]["reported_composite_score"] == 0.99


@pytest.mark.parametrize("mutation", ["signature", "payload", "bundle", "untrusted", "wrong_key"])
def test_receipt_tampering_and_untrusted_keys_fail(bundle, keys, mutation):
    private, trusted = keys
    receipt = issue_receipt(bundle, "independent", private)
    if mutation == "signature":
        receipt["signature"] = "00" * 64
    elif mutation == "payload":
        receipt["payload"]["audit"]["adjusted_score"] = 1
    elif mutation == "bundle":
        bundle["system_name"] = "another-system"
    elif mutation == "untrusted":
        trusted = {}
    else:
        receipt = issue_receipt(bundle, "independent", Ed25519PrivateKey.generate())
    with pytest.raises(ValueError):
        verify_receipt(bundle, receipt, trusted)


def test_even_a_trusted_signature_cannot_bypass_replay(bundle, keys):
    from rsi_bench.certification import _encode
    private, trusted = keys
    receipt = issue_receipt(bundle, "independent", private)
    receipt["payload"]["audit"]["adjusted_score"] = 1
    receipt["signature"] = private.sign(_encode(receipt["payload"])).hex()
    with pytest.raises(ValueError, match="does not match"):
        verify_receipt(bundle, receipt, trusted)


@pytest.mark.parametrize("mutation", ["source", "runtime", "nan", "unknown_axis", "cycles", "seed"])
def test_incompatible_or_invalid_evidence_fails_closed(bundle, mutation):
    if mutation == "source":
        bundle["implementation"]["sources_sha256"] = "0" * 64
    elif mutation == "runtime":
        bundle["implementation"]["scipy"] = "other"
    elif mutation == "nan":
        bundle["claims"]["composite_score"] = float("nan")
    elif mutation == "unknown_axis":
        bundle["protocol"]["axes"].append("unknown")
    elif mutation == "cycles":
        bundle["protocol"]["max_cycles"] = 0
    else:
        bundle["protocol"]["seed"] = True
    with pytest.raises(ValueError):
        recompute(bundle)


def test_json_failure_cannot_be_hidden_by_ssm_exception_handler():
    bench = benchmark(axes=["ssm"])
    original = bench.system
    original.modify_fn = lambda: {"bad": object()}
    with pytest.raises(ValueError, match="finite JSON"):
        record_run(bench, max_cycles=8)
    assert bench.system is original


def test_dictionary_order_is_bound_to_the_receipt(bundle, keys):
    private, trusted = keys
    receipt = issue_receipt(bundle, "independent", private)
    reordered = copy.deepcopy(bundle)
    reordered["events"][1]["value"] = dict(reversed(list(bundle["events"][1]["value"].items())))
    with pytest.raises(ValueError, match="does not match"):
        verify_receipt(reordered, receipt, trusted)


def test_custom_weights_rejected_and_ids_unique(bundle):
    bench = benchmark()
    bench.scorer.weights = {"ssm": 1}
    with pytest.raises(ValueError, match="default scoring"):
        record_run(bench)
    with pytest.raises(ValueError, match="unique"):
        rank_submissions([{"id": "same"}, {"id": "same"}], {})


def test_oversized_numeric_claim_is_a_mismatch(bundle):
    bundle["claims"]["composite_score"] = 10 ** 1000
    assert not recompute(bundle)["eligible"]


def test_signed_failed_audit_remains_visible_without_rank(bundle, keys):
    private, trusted = keys
    bundle["claims"]["axes"]["ssm"]["metrics"]["total_crashes"] = 100
    receipt = issue_receipt(bundle, "independent", private)
    rows = rank_submissions([{"id": "bad-claim", "bundle": bundle, "receipt": receipt}], trusted)
    assert not rows["ranked"]
    assert rows["unverifiable"][0]["non_recomputable_claims"] == ["ssm"]


def test_receipt_cannot_be_reused_for_a_different_protocol(bundle, keys):
    private, trusted = keys
    receipt = issue_receipt(bundle, "independent", private)
    bundle["protocol"]["seed"] = 999
    with pytest.raises(ValueError, match="does not match"):
        verify_receipt(bundle, receipt, trusted)
