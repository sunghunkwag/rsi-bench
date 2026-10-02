"""Local mechanics demo; the temporary signer is NOT an independent verifier."""
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from rsi_bench.certification import issue_receipt, rank_submissions, record_run
from rsi_bench.core import RSIBenchmark


class DemoSystem:
    def __init__(self):
        self.cycle = 0

    def modify(self):
        self.cycle += 1
        return {"level": min(self.cycle // 10, 4),
                "operators": [{"type": "demo", "cross_task": True}],
                "goals": [{"name": "goal-{}".format(self.cycle), "feasible": True}]}

    def evaluate(self):
        return {"fitness": 1 + self.cycle / 100}

    def state(self):
        return {"sandbox_violation": False}


def main():
    system = DemoSystem()
    bench = RSIBenchmark()
    bench.register_system("local-demo", system.modify, system.evaluate, system.state)
    _, bundle = record_run(bench, max_cycles=50, seed=42)
    # Kept in memory only. Production signing belongs to an independent party.
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    receipt = issue_receipt(bundle, "local-demo-signer", private)
    rows = rank_submissions([
        {"id": "demo-replay", "bundle": bundle, "receipt": receipt},
        {"id": "demo-self-report", "bundle": bundle},
    ], {"local-demo-signer": public})
    print("LOCAL DEMO ONLY: not independently measured or certified")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
