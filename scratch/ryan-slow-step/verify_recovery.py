"""Bounded recovery proof using the reporter's unchanged decks, seed and actions."""

import ast
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import managym
from manabot.env.observation import ObservationSpace


def main() -> None:
    source = Path(__file__).with_name("reporter_manabot_slow_step.py")
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SEED", "DECKS", "ACTIONS"}
    }
    players = [
        managym.PlayerConfig(f"seat {seat}", deck, world=managym.WORLD_SETUP, disclosure="closed")
        for seat, deck in enumerate(values["DECKS"])
    ]
    scalar = managym.Env(seed=values["SEED"])
    scalar.reset(players)
    vector = managym.VectorEnv(1, seed=values["SEED"])
    buffers = ObservationSpace().encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    vector.set_buffers(buffers)
    vector.reset_all_into_buffers(players)
    scalar_times: list[float] = []
    vector_times: list[float] = []
    digest = hashlib.sha256()
    search_root = None
    for number, action in enumerate(values["ACTIONS"], start=1):
        frame = scalar.semantic_decision_frame_json()
        assert vector.semantic_decision_frames_json() == [frame], number
        digest.update(frame.encode())
        if number == 426:
            search_root = scalar.clone_env()
        began = perf_counter()
        scalar.step(action)
        scalar_times.append(perf_counter() - began)
        began = perf_counter()
        vector.step_isolated_into_buffers([action], [True])
        vector_times.append(perf_counter() - began)
        assert "game_error" not in vector.get_last_info()[0], number
    frame = scalar.semantic_decision_frame_json()
    assert vector.semantic_decision_frames_json() == [frame]
    digest.update(frame.encode())
    print(json.dumps({
        "engine": managym.__file__, "steps": len(scalar_times),
        "scalar_final_ms": scalar_times[-1] * 1000,
        "scalar_max_ms": max(scalar_times) * 1000,
        "vector_final_ms": vector_times[-1] * 1000,
        "vector_max_ms": max(vector_times) * 1000,
        "decision_frames_sha256": digest.hexdigest(),
        "final_frame": json.loads(frame),
    }), flush=True)
    assert max(scalar_times + vector_times) < 0.25
    assert search_root is not None
    # Flat determinization requires open decklists. Replay the same actions in
    # that supported search setup; retain the closed scalar/vector proof above.
    open_root = managym.Env(seed=values["SEED"])
    open_root.reset([
        managym.PlayerConfig(f"seat {seat}", deck, world=managym.WORLD_SETUP, disclosure="open")
        for seat, deck in enumerate(values["DECKS"])
    ])
    for action in values["ACTIONS"][:-1]:
        open_root.step(action)
    assert json.loads(open_root.semantic_decision_frame_json())["offers"] == json.loads(search_root.semantic_decision_frame_json())["offers"]
    search_root = open_root
    before = search_root.state_digest()
    began = perf_counter()
    scores, simulations, caps = search_root.flat_mc_scores(1, 1, 42, 32)
    elapsed = perf_counter() - began
    assert search_root.state_digest() == before
    assert len(scores) == search_root.action_count()
    print(json.dumps({"search_s": elapsed, "scores": scores, "simulations": simulations, "caps": caps}), flush=True)


if __name__ == "__main__":
    main()
