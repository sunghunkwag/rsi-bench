"""End-to-end goal verification evidence and evaluation-version boundaries."""

import copy
import json

import pytest
import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rsi_bench.axes.axis6_goal_generation import GoalGeneration
from rsi_bench.certification import (
    issue_receipt, rank_submissions, recompute, record_run, verify_receipt,
)
from rsi_bench.core import RSIBenchmark


class ArithmeticSystem:
    def __init__(self, correct=True):
        self.cycle = 0
        self.correct = correct

    def modify(self):
        self.cycle += 1
        return {"level": min(self.cycle // 5, 4),
                "operators": [{"type": "arithmetic", "cross_task": True}],
                "goals": [{"left": self.cycle, "right": 1, "solved": True}]}

    def state(self):
        return {"answer": self.cycle + 1 if self.correct else -1,
                "sandbox_violation": False}

    def evaluate(self):
        return {"fitness": 1 + self.cycle / 100}


def goal_check(goal, state):
    expected = goal["left"] + goal["right"]
    return {"goal_id": "addition:{}:1".format(goal["left"]),
            "feasible": True, "solved": state["answer"] == expected,
            "complexity": min(expected / 100.0, 1.0)}


def benchmark(verifier=goal_check, correct=True, verifier_id="arithmetic-fixture-v1"):
    system = ArithmeticSystem(correct)
    bench = RSIBenchmark(goal_verifier=verifier, goal_verifier_id=verifier_id)
    bench.register_system("arithmetic", system.modify, system.evaluate, system.state)
    return bench


@pytest.fixture
def evidence():
    return record_run(benchmark(), max_cycles=8, seed=0)[1]


@pytest.fixture
def signer():
    key = Ed25519PrivateKey.generate()
    raw = key.public_key().public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw)
    return key, {"fixture-owner": raw}


def test_verified_and_dishonest_goals_replay_without_generic_fitness_credit():
    correct, bundle = record_run(benchmark(), max_cycles=8)
    wrong, wrong_bundle = record_run(benchmark(correct=False), max_cycles=8)
    assert correct.axis_results[GoalGeneration.name].score > 0
    assert wrong.axis_results[GoalGeneration.name].score == 0
    assert recompute(json.loads(json.dumps(bundle)))["eligible"]
    assert recompute(wrong_bundle)["eligible"]
    assert bundle["schema"] == "rsi-bench-evidence-v2"
    assert bundle["protocol"]["evaluation_protocol"] == "rsi-bench-evaluation-v2"
    calls = [e for e in bundle["events"] if e["method"] == "verify_goal"]
    assert len(calls) == 8
    assert all(len(e["args"]) == 2 for e in calls)


def test_recorded_and_ordinary_scores_match_and_original_verifier_is_restored():
    bench = benchmark()
    original_system, original_verifier = bench.system, bench.goal_verifier
    ordinary = benchmark().run(max_cycles=8, seed=0, verbose=False)
    recorded, _ = record_run(bench, max_cycles=8, seed=0)
    assert recorded.composite_score == ordinary.composite_score
    assert recorded.axis_results[GoalGeneration.name].metrics == ordinary.axis_results[GoalGeneration.name].metrics
    assert bench.system is original_system
    assert bench.goal_verifier is original_verifier


def test_replay_and_receipt_validation_never_reexecute_original_verifier(signer):
    calls = []

    def counting_verifier(goal, state):
        calls.append(goal)
        return goal_check(goal, state)

    _, bundle = record_run(benchmark(verifier=counting_verifier), max_cycles=8)
    original_calls = len(calls)
    private, trusted = signer
    receipt = issue_receipt(bundle, "fixture-owner", private)
    assert verify_receipt(bundle, receipt, trusted)["eligible"]
    assert len(calls) == original_calls == 8
    assert receipt["payload"]["schema"] == "rsi-bench-receipt-v2"


def test_verifier_errors_are_recorded_and_replayed_as_unverified_goals():
    def unavailable(goal, state):
        raise RuntimeError("measurement unavailable")

    results, bundle = record_run(benchmark(verifier=unavailable), max_cycles=8)
    goal_events = [e for e in bundle["events"] if e["method"] == "verify_goal"]
    assert all(e["error"] is True and len(e["args"]) == 2 for e in goal_events)
    assert results.axis_results[GoalGeneration.name].score == 0
    assert results.axis_results[GoalGeneration.name].metrics["verification_failures"] == 8
    assert recompute(bundle)["eligible"]


def test_callback_argument_snapshots_precede_verifier_mutations():
    def mutating_check(goal, state):
        verdict = goal_check(goal, state)
        goal["left"] = -100
        state["answer"] = -100
        return verdict

    _, bundle = record_run(benchmark(verifier=mutating_check), max_cycles=8)
    first = next(e for e in bundle["events"] if e["method"] == "verify_goal")
    assert first["args"][0]["left"] > 0
    assert first["args"][1]["answer"] > 0
    assert recompute(bundle)["eligible"]


@pytest.mark.parametrize("mutation", ["goal", "state", "missing_args", "extra_args", "availability"])
def test_verifier_argument_and_availability_tampering_fails_closed(evidence, mutation):
    event = next(e for e in evidence["events"] if e["method"] == "verify_goal")
    if mutation == "goal":
        event["args"][0]["left"] = 999
    elif mutation == "state":
        event["args"][1]["answer"] = 999
    elif mutation == "missing_args":
        del event["args"]
    elif mutation == "extra_args":
        event["args"].append(None)
    else:
        evidence["protocol"]["goal_verifier_id"] = None
    with pytest.raises(ValueError):
        recompute(evidence)


def test_changed_verdict_cannot_retain_old_claims_or_receipt(evidence, signer):
    private, trusted = signer
    receipt = issue_receipt(evidence, "fixture-owner", private)
    event = next(e for e in evidence["events"] if e["method"] == "verify_goal")
    event["value"]["solved"] = False
    report = recompute(evidence)
    assert not report["eligible"]
    assert "agg" in report["non_recomputable_claims"]
    with pytest.raises(ValueError):
        verify_receipt(evidence, receipt, trusted)


@pytest.mark.parametrize("field,value", [
    ("schema", "rsi-bench-evidence-v1"),
    ("evaluation_protocol", "rsi-bench-evaluation-v1"),
])
def test_v1_evidence_and_scores_are_not_silently_upgraded(evidence, field, value):
    if field == "schema":
        evidence[field] = value
    else:
        evidence["protocol"][field] = value
    with pytest.raises(ValueError):
        recompute(evidence)


def test_goal_fixture_identity_is_part_of_ranking_policy(evidence, signer):
    private, trusted = signer
    receipt = issue_receipt(evidence, "fixture-owner", private)
    submission = {"id": "checked-goals", "bundle": evidence, "receipt": receipt}
    matching = rank_submissions([submission], trusted, protocol=evidence["protocol"])
    assert matching["ranked"][0]["rank"] == 1
    policy = copy.deepcopy(evidence["protocol"])
    policy["goal_verifier_id"] = "different-fixture-v1"
    mismatching = rank_submissions([submission], trusted, protocol=policy)
    assert not mismatching["ranked"]
    assert "ranking protocol" in mismatching["unverifiable"][0]["reason"]


def test_old_ranking_protocol_is_rejected(evidence, signer):
    _, trusted = signer
    with pytest.raises(ValueError, match="ranking protocol"):
        rank_submissions([], trusted, protocol={"axes": ["agg"], "max_cycles": 8, "seed": 0})


def test_nonfinite_verifier_output_cannot_be_hidden_as_a_callback_failure():
    def invalid_check(goal, state):
        return dict(goal_check(goal, state), complexity=float("nan"))

    bench = benchmark(verifier=invalid_check)
    original = bench.goal_verifier
    with pytest.raises(ValueError, match="finite JSON"):
        record_run(bench, max_cycles=8)
    assert bench.goal_verifier is original


@pytest.mark.parametrize("field,value", [("solved", np.bool_(True)),
                                         ("complexity", np.float64(0.5))])
def test_recording_cannot_normalize_invalid_raw_verdicts_into_success(field, value):
    def invalid_check(goal, state):
        return dict(goal_check(goal, state), **{field: value})

    ordinary = benchmark(verifier=invalid_check).run(max_cycles=8, verbose=False)
    recorded, bundle = record_run(benchmark(verifier=invalid_check), max_cycles=8)
    assert ordinary.axis_results[GoalGeneration.name].score == 0
    assert recorded.axis_results[GoalGeneration.name].metrics == ordinary.axis_results[GoalGeneration.name].metrics
    assert recompute(bundle)["eligible"]


def test_cyclic_verifier_output_fails_recording_and_restores_callback():
    def cyclic_check(goal, state):
        value = goal_check(goal, state)
        value["cycle"] = value
        return value

    bench = benchmark(verifier=cyclic_check)
    original = bench.goal_verifier
    with pytest.raises(ValueError, match="finite JSON"):
        record_run(bench, max_cycles=8)
    assert bench.goal_verifier is original
