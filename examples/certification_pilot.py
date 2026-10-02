"""Local mechanics demo; the temporary signer is NOT an independent verifier."""
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rsi_bench.certification import issue_receipt, rank_submissions, record_run
from rsi_bench.core import EVALUATION_PROTOCOL, RSIBenchmark

from verified_goal_generation import (
    ArithmeticGoalSystem, GOAL_VERIFIER_ID, verify_arithmetic_goal,
)


class DemoSystem(ArithmeticGoalSystem):
    def modify(self):
        result = super().modify()
        result.update({"level": min(self.cycle // 10, 4),
                       "operators": [{"type": "demo", "cross_task": True}]})
        return result

    def evaluate(self):
        return {"fitness": 1 + self.cycle / 100}


def main():
    system = DemoSystem()
    bench = RSIBenchmark(goal_verifier=verify_arithmetic_goal,
                         goal_verifier_id=GOAL_VERIFIER_ID)
    bench.register_system("local-demo", system.modify, system.evaluate, system.state)
    _, bundle = record_run(bench, max_cycles=50, seed=42)
    # Kept in memory only. Production signing belongs to an independent party.
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    receipt = issue_receipt(bundle, "local-demo-signer", private)
    # The deployment chooses the fixture cohort independently of submissions.
    protocol = {"axes": ["smd", "itq", "odr", "mas", "ssm", "agg"],
                "max_cycles": 50, "seed": 42,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "goal_verifier_id": GOAL_VERIFIER_ID}
    rows = rank_submissions([
        {"id": "demo-replay", "bundle": bundle, "receipt": receipt},
        {"id": "demo-self-report", "bundle": bundle},
    ], {"local-demo-signer": public}, protocol=protocol)
    assert len(rows["ranked"]) == 1
    assert len(rows["unverifiable"]) == 1
    print("LOCAL DEMO ONLY: not independently measured or certified")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
