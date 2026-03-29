"""Distribution Shift Tasks for RSI-Bench."""
import numpy as np
from rsi_bench.tasks import TaskBase


class DistributionShiftTask(TaskBase):
    """Evaluate adaptation under distribution shifts."""

    name = "distribution_shift"

    def __init__(self, n_phases=3, samples_per_phase=100, shift_magnitude=2.0):
        self.n_phases = n_phases
        self.samples_per_phase = samples_per_phase
        self.shift_magnitude = shift_magnitude
        self._phases = self._generate_phases()

    def _generate_phases(self):
        phases = []
        mean = np.zeros(2)
        for i in range(self.n_phases):
            shift = np.random.randn(2) * self.shift_magnitude
            mean = mean + shift
            cov = np.eye(2) * (1.0 + 0.5 * i)
            data = np.random.multivariate_normal(mean, cov, self.samples_per_phase)
            labels = (data[:, 0] + data[:, 1] > np.sum(mean)).astype(int)
            phases.append({"data": data.tolist(), "labels": labels.tolist(), "mean": mean.tolist()})
        return phases

    def get_initial_state(self):
        return {"current_phase": 0, "phase_scores": [], "predictions": []}

    def evaluate_candidate(self, state, predictor_fn):
        phase_idx = state["current_phase"]
        if phase_idx >= self.n_phases:
            return {"error": "all_phases_complete", "accuracy": 0.0}
        phase = self._phases[phase_idx]
        data = np.array(phase["data"])
        labels = np.array(phase["labels"])
        preds = predictor_fn(data)
        accuracy = float(np.mean(np.array(preds) == labels))
        state["phase_scores"].append(accuracy)
        state["current_phase"] += 1
        return {"accuracy": accuracy, "phase": phase_idx,
                "phases_remaining": self.n_phases - state["current_phase"]}

    def score(self, state):
        if not state["phase_scores"]:
            return 0.0
        scores = state["phase_scores"]
        base = np.mean(scores)
        recovery = np.mean([max(0, scores[i] - scores[i - 1])
                            for i in range(1, len(scores))]) if len(scores) > 1 else 0
        return float(0.7 * base + 0.3 * recovery)

    def generate(self, seed=None):
        return self._phases, None

    def evaluate(self, system_output, expected):
        return float(np.mean(np.array(system_output) == np.array(expected)))
