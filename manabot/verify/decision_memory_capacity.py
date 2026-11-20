"""Zero-cost Elemental heuristic census of complete viewer decision tables.

Native projections and encoding own all counts. ScalarCapture retains exact
choices for replay; no labels, fitted model or rental is involved. Terminal
prefixes include every retained decision for both viewers.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Literal

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
import managym
from managym.recording import ScalarCapture


@dataclass(frozen=True)
class CapacityGame:
    seed: int
    pair: tuple[str, str]
    disclosure: str
    status: str
    decisions: int
    counts: list[dict[str, int]]
    actions: list[int]
    error: str | None


def run_census(out: Path, max_decisions: int = 2000) -> None:
    """Retain incomplete games too; never recommend capacities after overflow.

    All 200 cells cover every ordered pair in both disclosure modes; no
    strength inference.
    """
    games = 200
    if max_decisions <= 0:
        raise ValueError("census requires positive game and decision bounds")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = ObservationConfig(
        max_decision_records=4096,
        max_decision_reference_details=131072,
        max_decision_references=4096,
        max_decision_members=131072,
    )
    space = ObservationSpace(config)
    buffers = space.encoder.allocate()
    keys = managym.authored_deck_keys("elemental-world-v2")
    if len(keys) != 10:
        raise ValueError("census requires ten Elemental decks")
    maxima = dict.fromkeys(MEMORY_FIELDS, 0)
    matchups: dict[str, dict[str, int]] = {}
    failures = 0
    for number in range(games):
        seed = 58100 + number
        pair = (keys[(number // 2) % 10], keys[(number // 20) % 10])
        disclosure: Literal["open", "closed"] = "open" if number % 2 == 0 else "closed"
        env = managym.Env(seed=seed, decision_memory=True)
        capture = ScalarCapture(
            seed, decision_memory=True, failure_directory=out / "failures"
        )
        obs, _ = capture.reset(
            env,
            managym.GameSetup(
                [
                    managym.authored_deck_setup("elemental-world-v2", key)
                    for key in pair
                ],
                decklists=disclosure,
            ),
        )
        selectors = (managym.Heuristic(), managym.Heuristic())
        decisions = 0
        while not obs.game_over and decisions < max_decisions:
            viewer = env.current_agent_index()
            obs, _, _, _, _ = capture.step(env, selectors[viewer].select_action(obs))
            decisions += 1
        status = "complete" if obs.game_over else "decision-cap"
        counts: list[dict[str, int]] = []
        error_text = None
        try:
            for viewer in range(2):
                managym.encode_observation_into(
                    env.observation_for_player(viewer), buffers
                )
                counts.append(
                    {
                        key: int(np.count_nonzero(np.any(buffers[key] != 0, axis=1)))
                        for key in MEMORY_FIELDS
                    }
                )
        except managym.ObservationCapacity as error:
            status, error_text = "capacity", str(error)
        if status != "complete":
            failures += 1
        matchup = " / ".join((*pair, disclosure))
        local = matchups.setdefault(matchup, dict.fromkeys(MEMORY_FIELDS, 0))
        for row in counts:
            for key, count in row.items():
                maxima[key] = max(maxima[key], count)
                local[key] = max(local[key], count)
        record = CapacityGame(
            seed,
            pair,
            disclosure,
            status,
            decisions,
            counts,
            list(capture.inputs.actions),
            error_text,
        )
        with (out / "games.jsonl").open("a") as stream:
            stream.write(json.dumps(asdict(record)) + "\n")
        print(
            f"census {seed} {matchup}: {status}, {decisions} decisions, {counts}",
            flush=True,
        )
    summary = {
        "games": games,
        "failures": failures,
        "maxima": maxima,
        "by_matchup": matchups,
        "recommended_capacities": None
        if failures
        else {"max_" + key: 1 << count.bit_length() for key, count in maxima.items()},
        "census_configuration": config.model_dump(),
        "launcher_cost_usd": 0,
        "seconds": time.monotonic() - started,
        "limits": "Heuristic terminal prefixes only; finite roster coverage, no learned-policy resource admission.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
