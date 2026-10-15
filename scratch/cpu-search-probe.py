"""Bounded calls to existing selected/reference PUCT, with unchanged roots."""

import cProfile
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from manabot.sim.mcts import determinized_puct
from manabot.sim.search_branch import REFERENCE_BRANCH_DRIVER_ID, SELECTED_BRANCH_DRIVER_ID
from manabot.sim.selected_branchdriver_teacher import _fresh_engine, runtime_identity


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = _fresh_engine(1197, 0)
    profiler = cProfile.Profile()
    rows: list[dict[str, object]] = []
    tape: list[int] = []
    for prefix in (0, 16, 48):
        while len(tape) < prefix:
            if root.is_game_over():
                raise RuntimeError("Fixture became terminal before requested prefix")
            action = root.random_action_index()
            root.step(action)
            tape.append(action)
        before = root.search_witness_json()
        selected = None
        for driver in (SELECTED_BRANCH_DRIVER_ID, REFERENCE_BRANCH_DRIVER_ID):
            start = time.perf_counter()
            if driver == SELECTED_BRANCH_DRIVER_ID:
                profiler.enable()
            result = determinized_puct(root, simulations=8, worlds=2, seed=1419 + prefix,
                                       max_steps=2000, branch_driver_id=driver,
                                       branch_match_id=f"cpu-probe-{prefix}")
            profiler.disable()
            elapsed = time.perf_counter() - start
            unchanged = root.search_witness_json() == before
            if not unchanged:
                raise RuntimeError("Search mutated source root")
            if selected is None:
                selected = result
            else:
                assert np.array_equal(selected.visit_counts, result.visit_counts)
                assert np.array_equal(selected.q_values, result.q_values)
                assert selected.root_value == result.root_value
                assert selected.cap_hits == result.cap_hits
                assert selected.tree_nodes == result.tree_nodes
            rows.append({"prefix": prefix, "tape": list(tape), "driver": driver,
                         "root_witness": json.loads(before), "root_unchanged": unchanged,
                         "seconds": elapsed, "profiled": driver == SELECTED_BRANCH_DRIVER_ID,
                         "result": asdict(result)})
    profiler.dump_stats("scratch/cpu-probes/search.pstats")
    payload = {"runtime": runtime_identity(), "world": "w4", "checkpoint": None,
               "policy": "uniform priors, random structured leaves",
               "config": {"deal_seed": 1197, "worlds": 2, "traversals": 8,
                          "max_steps": 2000, "device": "cpu", "threads": 1},
               "selected_reference_values_equal": True, "rows": rows,
               "limits": ["Three roots, no throughput confidence interval",
                          "Selected profiled, reference unprofiled: no speed ratio claim",
                          "No neural evaluator; current teacher boundary only"]}
    target = Path("scratch/cpu-probes/search.json")
    target.write_text(json.dumps(payload, default=_json_value, indent=2) + "\n")
    print(json.dumps({"roots": 3, "calls": 6, "parity": True,
                      "selected_seconds": sum(float(row["seconds"]) for row in rows if row["profiled"])}))


if __name__ == "__main__":
    main()
