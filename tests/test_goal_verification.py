"""Adversarial coverage for independently verified goal completion."""

from copy import deepcopy
import math

import numpy as np
import pytest

from rsi_bench.axes.axis6_goal_generation import GoalGeneration
from rsi_bench.core import RSIBenchmark, SystemInterface


class ScriptedGoalSystem:
    """Expose goal claims and observed outputs without validating either."""

    def __init__(self, states, modifications=None, performance=1_000_000.0):
        self.states = states
        self.modifications = modifications or [{} for _ in states]
        self.performance = performance
        self.cycle = -1

    def modify(self):
        self.cycle += 1
        return self.modifications[self.cycle]

    def evaluate(self):
        return {"fitness": self.performance, "score": self.performance}

    def get_state(self):
        return self.states[self.cycle]

    def as_interface(self):
        return SystemInterface(
            name="Untrusted goal producer",
            modify_fn=self.modify,
            evaluate_fn=self.evaluate,
            get_state_fn=self.get_state,
        )


def addition_goal(left=2, right=3, **claims):
    return {"operation": "addition", "operands": [left, right], **claims}


def addition_goal_id(goal):
    left, right = goal["operands"]
    return "addition:{}:{}".format(left, right)


def arithmetic_verifier(goal, state):
    """Check a concrete output against the harness's arithmetic objective."""
    left, right = goal["operands"]
    goal_id = addition_goal_id(goal)
    feasible = goal["operation"] == "addition" and left >= 0 and right >= 0
    observed_answer = state.get("answers", {}).get(goal_id)
    return {
        "goal_id": goal_id,
        "feasible": feasible,
        "solved": feasible and observed_answer == left + right,
        "complexity": min((left + right) / 100.0, 1.0),
    }


def evaluate(states, modifications=None, verifier=arithmetic_verifier):
    system = ScriptedGoalSystem(states, modifications)
    result = GoalGeneration(goal_verifier=verifier).evaluate(
        system.as_interface(), max_cycles=len(states)
    )
    return result, system


def valid_verdict(**overrides):
    return {
        "goal_id": "trusted-objective",
        "feasible": True,
        "solved": True,
        "complexity": 0.5,
        **overrides,
    }


def assert_no_completion_credit(result):
    assert result.score == 0.0
    assert result.metrics["solve_rate"] == 0.0
    assert math.isfinite(result.score)


def test_actual_arithmetic_completion_is_a_positive_control():
    goal = addition_goal()
    result, _ = evaluate([
        {"goals": [goal], "answers": {addition_goal_id(goal): 5}},
    ])

    assert result.score > 0.0
    assert result.metrics["solve_rate"] == 1.0
    assert result.metrics["solved_goals"] == 1
    assert result.details["goal_complexities"] == [0.05]


@pytest.mark.parametrize("observed_answer", [None, -1, 4, 100_000])
def test_positive_general_fitness_does_not_prove_goal_completion(observed_answer):
    goal = addition_goal(feasible=True, solved=True, complexity=1_000_000)
    result, _ = evaluate([
        {"goals": [goal], "answers": {addition_goal_id(goal): observed_answer}},
    ])

    assert_no_completion_credit(result)


def test_without_a_trusted_verifier_even_correct_claims_receive_zero_credit():
    goal = addition_goal(feasible=True, solved=True, complexity=1.0)
    result, _ = evaluate([
        {"goals": [goal], "answers": {addition_goal_id(goal): 5}},
    ], verifier=None)

    assert_no_completion_credit(result)


@pytest.mark.parametrize("state_key", ["goals", "objectives", "generated_tasks"])
def test_explicit_state_goal_sources_are_verified(state_key):
    goal = addition_goal()
    result, _ = evaluate([
        {state_key: [goal], "answers": {addition_goal_id(goal): 5}},
    ])

    assert result.metrics["solved_goals"] == 1
    assert result.score > 0.0


def test_explicit_modification_goals_are_verified():
    goal = addition_goal()
    result, _ = evaluate(
        [{"answers": {addition_goal_id(goal): 5}}],
        [{"goals": [goal]}],
    )

    assert result.metrics["solved_goals"] == 1
    assert result.score > 0.0


@pytest.mark.parametrize("modification", [
    None,
    "A successful modification",
    {"level": 4, "fitness": 100_000, "feasible": True, "solved": True},
])
def test_modifications_without_goals_do_not_create_implicit_objectives(modification):
    calls = []

    def verifier(goal, state):
        calls.append(goal)
        return valid_verdict()

    result, _ = evaluate(
        [{"answers": {"unrelated": 5}}],
        [modification],
        verifier=verifier,
    )

    assert calls == []
    assert result.metrics["total_goals"] == 0
    assert_no_completion_credit(result)


@pytest.mark.parametrize("invalid_collection", ["claimed goal", {"solved": True}, None])
def test_malformed_goal_collections_do_not_create_goals(invalid_collection):
    calls = []

    def verifier(goal, state):
        calls.append(goal)
        return valid_verdict()

    result, _ = evaluate(
        [{field: invalid_collection for field in ("goals", "objectives", "generated_tasks")}],
        [{"goals": invalid_collection}],
        verifier=verifier,
    )

    assert calls == []
    assert result.metrics["total_goals"] == 0
    assert_no_completion_credit(result)


def test_tuple_goal_collections_are_supported_without_allowing_tuple_payloads():
    goal = addition_goal()
    result, _ = evaluate(
        [{"goals": (goal,), "answers": {addition_goal_id(goal): 5}}],
        [{"goals": (goal,)}],
    )

    assert result.metrics["total_goals"] == 1
    assert result.metrics["solved_goals"] == 1
    assert result.score > 0.0


@pytest.mark.parametrize("malformed_goal", [
    {1: "A nonstring goal key"},
    {"nested": {1: "A nested nonstring goal key"}},
    {"operation": "addition", "operands": (2, 3)},
    {"nested": [("A nested tuple",)]},
    {"nested": {"A set value"}},
    {"nested": float("nan")},
    {"nested": float("inf")},
    {"nested": object()},
])
def test_non_json_goal_payloads_fail_closed_before_verification(malformed_goal):
    calls = []

    def verifier(goal, state):
        calls.append(goal)
        return valid_verdict()

    result, _ = evaluate([{"goals": [malformed_goal]}], verifier=verifier)

    assert calls == []
    assert_no_completion_credit(result)


def test_cyclic_goal_payload_fails_closed_before_verification():
    cyclic_goal = {"name": "Cyclic proposal"}
    cyclic_goal["self"] = cyclic_goal
    calls = []

    def verifier(goal, state):
        calls.append(goal)
        return valid_verdict()

    result, _ = evaluate([{"goals": [cyclic_goal]}], verifier=verifier)

    assert calls == []
    assert_no_completion_credit(result)


def test_excessively_nested_goal_payload_fails_closed_instead_of_crashing():
    nested_goal = "Deeply nested proposal"
    for _ in range(1_100):
        nested_goal = {"nested": nested_goal}
    calls = []

    def verifier(goal, state):
        calls.append(goal)
        return valid_verdict()

    result, _ = evaluate([{"goals": [nested_goal]}], verifier=verifier)

    assert calls == []
    assert_no_completion_credit(result)


def test_canonical_duplicates_across_all_sources_are_verified_once_per_cycle():
    goal = addition_goal()
    reordered_goal = {"operands": [2, 3], "operation": "addition"}
    calls = []

    def verifier(proposal, state):
        calls.append(proposal)
        return arithmetic_verifier(proposal, state)

    result, _ = evaluate(
        [{"goals": [goal, reordered_goal], "objectives": [reordered_goal],
          "generated_tasks": [goal], "answers": {addition_goal_id(goal): 5}}],
        [{"goals": [reordered_goal]}],
        verifier=verifier,
    )

    assert len(calls) == 1
    assert result.metrics["total_goals"] == 1
    assert result.metrics["unique_goals"] == 1
    assert result.metrics["solved_goals"] == 1


def test_unresolved_goal_can_be_verified_after_observed_completion():
    goal = addition_goal()
    calls = []

    def verifier(proposal, state):
        calls.append(state["answers"][addition_goal_id(proposal)])
        return arithmetic_verifier(proposal, state)

    result, _ = evaluate([
        {"goals": [goal], "answers": {addition_goal_id(goal): 0}},
        {"goals": [deepcopy(goal)], "answers": {addition_goal_id(goal): 5}},
        {"goals": [deepcopy(goal)], "answers": {addition_goal_id(goal): 5}},
    ], verifier=verifier)

    assert calls == [0, 5]
    assert result.metrics["solved_goals"] == 1
    assert result.score > 0.0


def test_aliases_with_the_same_trusted_identity_only_receive_one_solved_credit():
    first = addition_goal(label="First wording")
    second = addition_goal(label="Different wording", solved=True)
    baseline, _ = evaluate([
        {"goals": [addition_goal()], "answers": {addition_goal_id(first): 5}}
        for _ in range(2)
    ])
    result, _ = evaluate([
        {"goals": [first, second], "answers": {addition_goal_id(first): 5}},
        {"goals": [deepcopy(second)], "answers": {addition_goal_id(first): 5}},
    ])

    assert result.metrics["solved_goals"] == 1
    assert result.score == pytest.approx(baseline.score)
    assert result.metrics == baseline.metrics
    assert result.details["goal_complexities"] == baseline.details["goal_complexities"]


def test_padding_aliases_cannot_manufacture_a_curriculum_over_multiple_cycles():
    states = []
    for index in range(6):
        goal = addition_goal(description="Claimed difficulty " * (10 ** index),
                             complexity=index / 5.0)
        states.append({"goals": [goal], "answers": {addition_goal_id(goal): 5}})
    baseline, _ = evaluate([
        {"goals": [addition_goal()], "answers": {addition_goal_id(addition_goal()): 5}}
        for _ in range(len(states))
    ])

    result, _ = evaluate(states)

    assert result.score == pytest.approx(baseline.score)
    assert result.metrics == baseline.metrics
    assert result.details["goal_complexities"] == [0.05]
    assert result.metrics["curriculum_monotonicity"] == 0.0


def test_one_repeated_identity_cannot_claim_the_novelty_of_fifty_distinct_goals():
    repeated_goal = addition_goal(0, 50)
    repeated_states = [
        {"goals": [repeated_goal], "answers": {addition_goal_id(repeated_goal): 50}}
        for _ in range(50)
    ]
    distinct_states = []
    for left in range(50):
        goal = addition_goal(left, 50 - left)
        distinct_states.append({"goals": [goal], "answers": {addition_goal_id(goal): 50}})

    repeated, _ = evaluate(repeated_states)
    distinct, _ = evaluate(distinct_states)

    assert repeated.metrics["solved_goals"] == 1
    assert distinct.metrics["solved_goals"] == 50
    assert repeated.metrics["goal_novelty"] == pytest.approx(1 / 50)
    assert distinct.metrics["goal_novelty"] == 1.0
    assert repeated.score < distinct.score
    assert repeated.details["goal_complexities"] == [0.5]
    assert distinct.details["goal_complexities"] == [0.5] * 50


def test_goal_padding_and_self_reported_difficulty_do_not_increase_credit():
    ordinary = addition_goal()
    padded = addition_goal(
        description="Unsubstantiated complexity " * 10_000,
        complexity=1_000_000_000,
        feasible=False,
        solved=False,
    )
    baseline, _ = evaluate([
        {"goals": [ordinary], "answers": {addition_goal_id(ordinary): 5}},
    ])
    inflated, _ = evaluate([
        {"goals": [padded], "answers": {addition_goal_id(padded): 5}},
    ])

    assert inflated.score == pytest.approx(baseline.score)
    assert inflated.metrics == baseline.metrics
    assert inflated.details["goal_complexities"] == baseline.details["goal_complexities"]


def test_curriculum_uses_trusted_task_difficulty_instead_of_goal_text():
    states = []
    expected_complexities = []
    for index, right in enumerate([10, 20, 30, 40]):
        goal = addition_goal(
            left=0, right=right,
            complexity=1_000_000 - index,
            description="Padding " * (100 - index),
        )
        states.append({"goals": [goal], "answers": {addition_goal_id(goal): right}})
        expected_complexities.append(right / 100.0)

    result, _ = evaluate(states)

    assert result.details["goal_complexities"] == expected_complexities
    assert result.metrics["curriculum_monotonicity"] == 1.0


@pytest.mark.parametrize("failure", [RuntimeError("offline"), ValueError("bad output")])
def test_verifier_errors_do_not_award_completion_credit(failure):
    def verifier(goal, state):
        raise failure

    result, _ = evaluate([{"goals": [addition_goal()]}], verifier=verifier)

    assert_no_completion_credit(result)


def test_a_failed_verification_counts_against_credit_without_hiding_other_results():
    incomplete = addition_goal(1, 1)
    complete = addition_goal(2, 3)

    def verifier(goal, state):
        if goal["operands"] == [1, 1]:
            raise RuntimeError("Unavailable observation")
        return arithmetic_verifier(goal, state)

    result, _ = evaluate([
        {"goals": [incomplete, complete], "answers": {addition_goal_id(complete): 5}},
    ], verifier=verifier)

    assert result.score > 0.0
    assert result.metrics["solved_goals"] == 1
    assert result.metrics["total_goals"] == 2
    assert result.metrics["solve_rate"] == 0.5
    assert result.metrics["verification_failures"] == 1


class VerdictSubclass(dict):
    """Mapping subclasses are outside the strict verifier result contract."""


INVALID_VERDICTS = [
    None,
    True,
    [],
    "solved",
    VerdictSubclass(valid_verdict()),
    {},
    {"goal_id": "missing-fields"},
    valid_verdict(extra_field="unrecognized"),
    valid_verdict(goal_id=""),
    valid_verdict(goal_id="   "),
    valid_verdict(goal_id=1),
    valid_verdict(feasible=1),
    valid_verdict(feasible="yes"),
    valid_verdict(feasible=np.bool_(True)),
    valid_verdict(solved=1),
    valid_verdict(solved="true"),
    valid_verdict(solved=np.bool_(True)),
    valid_verdict(feasible=False, solved=True),
    valid_verdict(complexity=True),
    valid_verdict(complexity="0.5"),
    valid_verdict(complexity=None),
    valid_verdict(complexity=-0.01),
    valid_verdict(complexity=1.01),
    valid_verdict(complexity=float("nan")),
    valid_verdict(complexity=float("inf")),
    valid_verdict(complexity=float("-inf")),
]


@pytest.mark.parametrize("verdict", INVALID_VERDICTS)
def test_malformed_verdicts_fail_closed(verdict):
    result, _ = evaluate(
        [{"goals": [addition_goal()]}],
        verifier=lambda goal, state: verdict,
    )

    assert_no_completion_credit(result)


@pytest.mark.parametrize("complexity", [0, 1, 0.0, 1.0])
def test_verified_difficulty_bounds_accept_finite_numbers(complexity):
    result, _ = evaluate(
        [{"goals": [addition_goal()]}],
        verifier=lambda goal, state: valid_verdict(complexity=complexity),
    )

    assert result.metrics["solved_goals"] == 1
    assert result.score > 0.0
    assert result.details["goal_complexities"] == [complexity]


def test_verifier_mutation_cannot_change_original_goal_or_observed_state():
    goal = addition_goal()
    state = {"goals": [goal], "answers": {addition_goal_id(goal): 5},
             "nested_observation": {"unchanged": [1, 2, 3]}}
    original = deepcopy(state)

    def verifier(proposal, observation):
        verdict = arithmetic_verifier(proposal, observation)
        proposal["operands"][0] = 999
        observation["answers"].clear()
        observation["nested_observation"]["unchanged"].append(999)
        return verdict

    result, system = evaluate([state], verifier=verifier)

    assert result.score > 0.0
    assert system.states[0] == original
    assert goal == original["goals"][0]


def register_fixture(benchmark, system, **metadata):
    benchmark.register_system(
        name="Arithmetic output fixture",
        modify_fn=system.modify,
        evaluate_fn=system.evaluate,
        get_state_fn=system.get_state,
        **metadata,
    )


@pytest.mark.parametrize("run_method", ["run", "run_single_axis"])
def test_benchmark_supplies_harness_verifier_to_both_execution_paths(run_method):
    goal = addition_goal()
    system = ScriptedGoalSystem([
        {"goals": [goal], "answers": {addition_goal_id(goal): 5}},
    ])
    benchmark = RSIBenchmark(
        axes=["agg"], goal_verifier=arithmetic_verifier,
        goal_verifier_id="trusted-fixture-v1",
    )
    register_fixture(benchmark, system)

    if run_method == "run":
        results = benchmark.run(max_cycles=1, verbose=False)
        result = results.axis_results[GoalGeneration.name]
        assert results.config["evaluation_protocol"] == "rsi-bench-evaluation-v2"
        assert results.config["goal_verifier_id"] == "trusted-fixture-v1"
    else:
        result = benchmark.run_single_axis("agg", max_cycles=1)

    assert result.score > 0.0
    assert result.metrics["solved_goals"] == 1


@pytest.mark.parametrize("arguments", [
    {"goal_verifier": arithmetic_verifier},
    {"goal_verifier_id": "trusted-fixture-v1"},
])
def test_benchmark_requires_verifier_and_identity_together(arguments):
    with pytest.raises((TypeError, ValueError)):
        RSIBenchmark(**arguments)


def test_system_metadata_cannot_supply_its_own_trusted_verifier():
    goal = addition_goal(feasible=True, solved=True)
    system = ScriptedGoalSystem([{"goals": [goal], "answers": {addition_goal_id(goal): 0}}])
    benchmark = RSIBenchmark(axes=["agg"])
    register_fixture(
        benchmark, system,
        goal_verifier=lambda goal, state: valid_verdict(),
        goal_verifier_id="self-reported-verifier",
    )

    results = benchmark.run(max_cycles=1, verbose=False)

    assert_no_completion_credit(results.axis_results[GoalGeneration.name])
    assert results.config.get("goal_verifier_id") != "self-reported-verifier"


def test_ordinary_benchmark_results_record_the_new_evaluation_protocol():
    system = ScriptedGoalSystem([{"goals": []}])
    benchmark = RSIBenchmark(axes=["agg"])
    register_fixture(benchmark, system)

    results = benchmark.run(max_cycles=1, verbose=False)

    assert results.config["evaluation_protocol"] == "rsi-bench-evaluation-v2"
    assert_no_completion_credit(results.axis_results[GoalGeneration.name])
