# Optional certification pilot

Issue [#1](https://github.com/sunghunkwag/rsi-bench/issues/1) proposes independent
recomputation, signed receipts, receipt-only rankings, and integrity penalties.
Reproducible evidence is useful, but RSI-Bench currently has no hosted leaderboard
or independent measurement service. This pilot provides protocol primitives.
Release 0.2.0 also introduces verified goal scoring through the
[AGG v2 contract](goal_verification.md); the composite scoring formula is unchanged.

## What v2 verifies

`record_run()` records every `modify`, `evaluate`, `state`, `reset`, and configured
`verify_goal` callback result in order, including callback failures. Goal verifier
events also include the copied `(goal, state)` arguments. `recompute()` feeds
those snapshots through the axis evaluators and scorer; it never executes the
submitted system or the original goal verifier. It checks all six axis scores
**and their complete metric dictionaries**,
plus the composite score. Duration is deliberately outside the claim set.

The evidence includes the system name, axes in execution order, cycle count,
seed, evaluation protocol, goal verifier fixture ID, reset availability, source
fingerprint, and Python/NumPy/SciPy versions.
Replay requires the same fingerprint and runtime versions. Archive the benchmark
revision and an environment lock alongside evidence. v2 supports the built-in
axes and default weights, finite JSON callback values, positive cycle counts up
to 10,000, and at most 1,000,000 callback events. Non-JSON/non-finite values cause
the recording to fail even if the safety axis catches the callback exception.

Callbacks are snapshotted by value, including NumPy scalar conversion. Record
mode supplies copies to the evaluators so later mutations cannot alter earlier
observations. Systems relying on shared object identity across callbacks are
outside this recording protocol. Ordinary `benchmark.run()` is not subject to
the recorder's JSON-only snapshot restriction.

AGG deduplicates proposals using canonical JSON and groups aliases by the trusted
verifier's stable goal identity. The evidence separately preserves dictionary
insertion order so exact callback snapshots and verifier arguments remain bound
to the digest. Save evidence using ordinary `json.dump`,
without `sort_keys=True`. Receipts use ASCII-escaped compact JSON with insertion
order preserved and SHA-256 over those encoded bytes. JSON whitespace changes
are harmless; reordering dictionaries invalidates an existing receipt.

## Trust boundary

An Ed25519 receipt authenticates **who attested to a particular replay audit**.
It does not establish that a recorded observation is true, that the callbacks
were independent, that an agent improved its code, or that a goal was solved.
A fabricated transcript can replay successfully, including fabricated goal
verifier verdicts. AGG v2 requires a benchmark-owned task verifier during a live
run, but replay consumes its recorded verdicts. The remaining axes still consume
system-provided metadata. `VERIFIED_REPLAY` is intentionally
a narrow label; it is not a safety or capability certification.

A deployment must configure an external allowlist of independent verifier IDs
and raw Ed25519 public keys. Public keys supplied by a submitter are not trusted
automatically. No verifier is pre-approved, including the proposer. The verifier
must control its signing key and use a clean benchmark environment. Before making
stronger claims, it must collect observations itself or reproduce measurements
against independent fixtures. Key rotation/revocation and admission policy belong
to the deployment; rerun validation against the current allowlist on every ranking.

v2 does not certify the proposer's four-piece issue/diff/trajectory/timestamp
bundle, consumption events, real distribution shifts, or autonomous behavior.
Those need explicit measurement adapters and adversarial fixtures, not a signature
over self-reported fields. No external pilot transcript has been certified here.

## Integrity and rankings

There are seven fixed claim groups: six axis score/metric groups and one composite.
A group is recomputable only if its complete submitted values match replay within
`1e-12` relative/absolute numeric tolerance. Missing groups also fail. Using a
fixed denominator prevents dilution by adding many trivial claims.

```
integrity_score = verified_claim_groups / 7
adjusted_score = max(0, recomputed_composite - (1 - integrity_score))
```

The adjusted score is a separate pilot policy, not a change to `UnifiedScorer`.
Partial or mismatching runs can receive a signed audit showing their failures,
but cannot enter the certification ranking. Only a complete six-axis replay with
seven matching groups and a trusted receipt receives a rank. Valid fully verified
runs have integrity 1, so their adjusted score equals the usual RSI score.

`rank_submissions()` returns `ranked` and `unverifiable` collections. Every
unsigned, invalid, incompatible, partial, or mismatching submission remains on
the `UNVERIFIABLE` wall with a reason and no rank. The label means insufficient
evidence for this track; it does not imply fraud. Legacy JSON exports can remain
visible there but cannot be certified without callback evidence. IDs must be
unique; ties are sorted by ID. This is a library API, not a hosted leaderboard.
All ranked runs must also use the same deployment-configured protocol, defaulting
to all six axes in standard order, 50 cycles, and seed 42. Supply `protocol=` to
choose a different comparison cohort; do not derive this policy from untrusted
submissions. The default cohort uses `rsi-bench-evaluation-v2` and no goal verifier
(`goal_verifier_id=None`), so its AGG score is zero. A configured goal verifier
requires an explicit matching `protocol=`. Its fixture ID must match as well as
axes, cycle budget, seed, and evaluation protocol. This ID does not prove trusted
measurement and is distinct from the receipt signer's `verifier_id`. Identical
evidence can enter a ranking only once.

Evidence and receipts use `rsi-bench-evidence-v2` and `rsi-bench-receipt-v2`. v1
evidence is rejected. Earlier AGG and composite scores are not comparable with
evaluation v2; rerun systems and collect new evidence under the chosen fixtures.

## Usage

Install the optional signing dependency:

```bash
pip install -e '.[certification]'
python examples/certification_pilot.py
```

For a registered benchmark:

```python
import json
from rsi_bench.certification import record_run, recompute, issue_receipt

results, bundle = record_run(bench, max_cycles=50, seed=42)
with open("evidence.json", "w", encoding="utf-8") as f:
    json.dump(bundle, f, indent=2)

# Independent verifier process, with its own private key:
audit = recompute(bundle)
receipt = issue_receipt(bundle, "approved-verifier", verifier_private_key)
```

At the ranking boundary:

```python
from rsi_bench.certification import rank_submissions
from rsi_bench.core import EVALUATION_PROTOCOL

rows = rank_submissions(
    [{"id": "run-001", "bundle": bundle, "receipt": receipt}],
    trusted_verifiers={"approved-verifier": approved_public_key_bytes},
    protocol={"axes": ["smd", "itq", "odr", "mas", "ssm", "agg"],
              "max_cycles": 50, "seed": 42,
              "evaluation_protocol": EVALUATION_PROTOCOL,
              "goal_verifier_id": "trusted-fixtures-v1"},
)
```

`verify_receipt()` validates the signature against that allowlist, then repeats
the audit and compares it to the signed payload. Swapped evidence, altered scores,
truncated/extra/reordered callbacks, altered verifier arguments, and mismatched
runtimes fail closed. A ranking protocol must contain exactly the five fields in
the example. Configure `bench` with the corresponding benchmark-owned verifier
and fixture ID before recording; use `goal_verifier_id=None` for a cohort without
goal verification.

The example uses a temporary local demo key to exercise the mechanics. It is not
an independent-verification result and should not be published as one.
