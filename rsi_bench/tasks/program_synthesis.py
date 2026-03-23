"""Program Synthesis Tasks for RSI-Bench."""
import numpy as np
from rsi_bench.tasks import TaskBase


class ProgramSynthesisTask(TaskBase):
      """Synthesize programs from input-output specifications."""

    name = "program_synthesis"

    def __init__(self, io_pairs=None, max_length=200):
              self.io_pairs = io_pairs or [
                            ([1, 2, 3], [2, 4, 6]),
                            ([0, 5, 10], [0, 10, 20]),
                            ([-1, -2], [-2, -4]),
              ]
              self.max_length = max_length

    def get_initial_state(self):
              return {"io_pairs": self.io_pairs, "best_program": None, "best_pass_rate": 0.0, "attempts": 0}

    def evaluate_candidate(self, state, program_fn):
              state["attempts"] += 1
              passed = 0
              for inp, expected in self.io_pairs:
                            try:
                                              result = program_fn(inp)
                                              if result == expected:
                                                                    passed += 1
                            except Exception:
                                              pass
                                      rate = passed / len(self.io_pairs)
                        if rate > state["best_pass_rate"]:
                                      state["best_pass_rate"] = rate
                                      state["best_program"] = str(program_fn)
                                  return {"pass_rate": rate, "passed": passed, "total": len(self.io_pairs)}

    def score(self, state):
              return state["best_pass_rate"]


PRESET_SPECS = {
      "double": [([1, 2, 3], [2, 4, 6]), ([0], [0]), ([-3], [-6])],
      "reverse_sort": [([3, 1, 2], [1, 2, 3]), ([5, 4], [4, 5])],
      "cumulative_sum": [([1, 2, 3], [1, 3, 6]), ([4, 1], [4, 5])],
}
