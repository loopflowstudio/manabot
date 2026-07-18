"""Conditional (belief-conditioned) distillation support — INT-14.

Extends the flat-MC decision shard (`manabot/sim/distill.py`) with a
per-decision condition axis so a policy/value student can be conditioned on a
provided per-condition strategy label carried by a `ConditionalStrategyResult`
(INT-13, `manabot/sim/conditional_search.py`).

The condition is a **provided side input**, not a predicted belief. There is no
learned belief head, range net, or per-hand value vector here; the policy and
scalar value heads of `Agent` are unchanged. The condition enters the student
as one neutral object row built from a per-row `condition_index` and
`condition_weight`.

Frozen shape contract
---------------------
`CONDITIONAL_STRATEGY_SHAPE` pins the viewer-safe shape of a
`ConditionalStrategyResult` (the layer `conditional_search.serialize_result`
exposes: `action_count`, `action_labels`, five per-condition
`condition_id`/`condition_mass`/`action_distribution`/`q_values`/`root_value`/
`uncertainty`/`simulations`). `conditional_strategy_shape_digest()` is the
SHA-256 over that pinned shape. `assert_conforms_to_conditional_strategy_shape`
validates any viewer-safe dict against it, so when INT-13's real
`ConditionalStrategyResult` is ingested it requires no contract change — a
divergence is a visible, reviewable schema bump, not a silent break.

The toy producer
----------------
`generate_uniform_determinization_shard` produces a conditional shard that
conforms to the pinned shape without running a real conditional search: it
reuses the existing flat-MC self-play path (`distill.generate_selfplay_shard`)
and emits K conditions per decision whose `condition_scores` are all the
flat-MC per-action scores (uniform determinization — every world yields the
same marginal strategy), with uniform weights 1/K. The `condition_index` tag
varies but carries no hidden information, so the pre-registered strength
prediction is ~0 gap. This is a plumbing/measurement receipt, not a strength
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from manabot.sim.distill import (
    META_KEYS,
    OBS_KEYS,
    SCORE_KEY,
    _git_commit,
    generate_selfplay_shard,
)

# Per-decision conditional shard keys (extend the flat-MC shard format).
CONDITION_COUNT_KEY = "condition_count"  # int16 (D,) — K for the decision
CONDITION_INDEX_KEY = "condition_index"  # int16 (D, K_max) — per-condition id; -1 pad
CONDITION_WEIGHT_KEY = "condition_weight"  # float32 (D, K_max) — belief weight; 0 pad
CONDITION_SCORES_KEY = "condition_scores"  # float32 (D, K_max, max_actions) — strategy
CONDITION_ROOT_VALUE_KEY = "condition_root_value"  # float32 (D, K_max) — optional
CONDITION_KEYS: tuple[str, ...] = (
    CONDITION_COUNT_KEY,
    CONDITION_INDEX_KEY,
    CONDITION_WEIGHT_KEY,
    CONDITION_SCORES_KEY,
)
# Per-row obs fields the loader attaches for the conditioned student. These are
# the side inputs `Agent._condition_row` reads; they are NOT part of the
# fixed ObservationSpace ABI (the arena path does not produce them).
CONDITION_ROW_KEYS: tuple[str, ...] = ("condition_index", "condition_weight")

# The five condition roles of an INT-13 ConditionalQueryPlan, in fixed order:
# True (unconditional baseline), Has, Lacks (= Not(Has)), Q, Not(Q). The
# per-row `condition_index` is the positional id 0..4 into this tuple. The
# human-readable condition_id strings are carried in shard provenance, not in
# the per-row training data.
CONDITION_ROLES: tuple[str, ...] = ("true", "has", "lacks", "q", "not_q")
DEFAULT_CONDITION_COUNT = len(CONDITION_ROLES)

CONDITION_LABEL_FORMAT = "uniform_determinization_world_index_v1"

# Pinned viewer-safe shape of a ConditionalStrategyResult (INT-13
# `serialize_result` output). The digest over this is the contract; any
# divergence is a reviewable schema bump.
CONDITIONAL_STRATEGY_SHAPE: dict[str, Any] = {
    "schema_version": 1,
    "planner": "determinized_puct",
    "top_level_fields": {
        "action_count": "int",
        "action_labels": "list[str]",
        "root_state_digest": "str",
        "search_params": "dict",
        "prior_sha256": "str",
        "plan_sha256": "str",
        "identities": "dict",
        "realized_compute": "dict",
        "comparison_deltas": "dict",
    },
    "conditions": {
        "count": 5,
        "roles": list(CONDITION_ROLES),
        "per_condition": {
            "condition_id": "str",
            "condition_mass": "float",
            "support": "int",
            "action_distribution": "list[float]",
            "q_values": "list[float]",
            "root_value": "float",
            "uncertainty": "float",
            "simulations": "int",
        },
    },
}


class ConditionalDistillError(ValueError):
    """Typed failure for conditional shard ingest/verify."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def conditional_strategy_shape_digest() -> str:
    """SHA-256 over the pinned ConditionalStrategyResult viewer-safe shape."""
    return _canonical_json_sha256(CONDITIONAL_STRATEGY_SHAPE)


def assert_conforms_to_conditional_strategy_shape(
    viewer_safe: Mapping[str, Any],
) -> None:
    """Fail closed when a viewer-safe dict diverges from the pinned shape.

    This is the ingest gate: INT-13's real `ConditionalStrategyResult`
    (serialized via `conditional_search.serialize_result`) must pass this
    unchanged. A divergence is a visible schema bump, not a silent break.
    """

    if not isinstance(viewer_safe, Mapping):
        raise ConditionalDistillError("viewer-safe payload must be a JSON object")
    if (
        int(viewer_safe.get("schema_version", -1))
        != CONDITIONAL_STRATEGY_SHAPE["schema_version"]
    ):
        raise ConditionalDistillError(
            "viewer-safe schema_version does not match the pinned contract"
        )
    if viewer_safe.get("planner") != CONDITIONAL_STRATEGY_SHAPE["planner"]:
        raise ConditionalDistillError(
            f"planner is {viewer_safe.get('planner')!r}, expected "
            f"{CONDITIONAL_STRATEGY_SHAPE['planner']!r}"
        )
    for field, kind in CONDITIONAL_STRATEGY_SHAPE["top_level_fields"].items():
        if field not in viewer_safe:
            raise ConditionalDistillError(f"viewer-safe payload missing {field!r}")
        if not _matches_kind(viewer_safe[field], kind):
            raise ConditionalDistillError(
                f"viewer-safe field {field!r} does not match kind {kind!r}"
            )
    conditions = viewer_safe.get("viewer_safe", {}).get("conditions")
    if not isinstance(conditions, list):
        raise ConditionalDistillError(
            "viewer-safe payload missing viewer_safe.conditions list"
        )
    expected_count = CONDITIONAL_STRATEGY_SHAPE["conditions"]["count"]
    if len(conditions) != expected_count:
        raise ConditionalDistillError(
            f"expected {expected_count} conditions, got {len(conditions)}"
        )
    per_condition = CONDITIONAL_STRATEGY_SHAPE["conditions"]["per_condition"]
    for i, cond in enumerate(conditions):
        if not isinstance(cond, Mapping):
            raise ConditionalDistillError(f"condition {i} is not an object")
        for field, kind in per_condition.items():
            if field not in cond:
                raise ConditionalDistillError(f"condition {i} missing {field!r}")
            if not _matches_kind(cond[field], kind):
                raise ConditionalDistillError(
                    f"condition {i} field {field!r} does not match kind {kind!r}"
                )
        action_count = int(viewer_safe["action_count"])
        for vec_field in ("action_distribution", "q_values"):
            if len(cond[vec_field]) != action_count:
                raise ConditionalDistillError(
                    f"condition {i} {vec_field} length {len(cond[vec_field])} "
                    f"!= action_count {action_count}"
                )


def _matches_kind(value: Any, kind: str) -> bool:
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str)
    if kind == "dict":
        return isinstance(value, Mapping)
    if kind == "list[str]":
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    if kind == "list[float]":
        return isinstance(value, list) and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in value
        )
    raise ConditionalDistillError(f"unknown kind {kind!r}")


# ---------------------------------------------------------------------------
# Adapter: viewer-safe ConditionalStrategyResult -> per-decision condition rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionRow:
    """One condition's strategy at one decision, in shard column order."""

    condition_index: int
    condition_weight: float
    condition_scores: np.ndarray  # (max_actions,) float32, -1 on invalid
    condition_root_value: float | None


def viewer_safe_to_condition_rows(
    viewer_safe: Mapping[str, Any],
    *,
    actions_valid: np.ndarray | None = None,
) -> list[ConditionRow]:
    """Convert one viewer-safe ConditionalStrategyResult to condition rows.

    The per-condition `q_values` (win-probability estimates in [0,1]) become
    the shard's `condition_scores`, with -1 padding on invalid actions. When
    ``actions_valid`` is omitted, all `action_count` actions are assumed legal
    (true for the INT-13 fixture's root). Weights are taken from
    `condition_mass` and normalized to sum to 1 over the conditions.

    This is the ingest path INT-13's real result will use; the shape is
    assert-checked first.
    """

    assert_conforms_to_conditional_strategy_shape(viewer_safe)
    action_count = int(viewer_safe["action_count"])
    conditions = viewer_safe["viewer_safe"]["conditions"]
    if actions_valid is None:
        actions_valid = np.ones(action_count, dtype=bool)
    else:
        actions_valid = np.asarray(actions_valid, dtype=bool)
        if actions_valid.shape != (action_count,):
            raise ConditionalDistillError(
                "actions_valid must have shape (action_count,)"
            )
    masses = np.asarray(
        [float(c["condition_mass"]) for c in conditions], dtype=np.float32
    )
    total = float(masses.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ConditionalDistillError("condition masses must have positive mass")
    weights = masses / total

    rows: list[ConditionRow] = []
    for k, cond in enumerate(conditions):
        scores = np.full(action_count, -1.0, dtype=np.float32)
        q = np.asarray(cond["q_values"], dtype=np.float32)
        legal = actions_valid
        if len(q) != action_count:
            raise ConditionalDistillError(
                f"condition {k} q_values length {len(q)} != action_count {action_count}"
            )
        scores[legal] = np.clip(q[legal], 0.0, 1.0)
        rows.append(
            ConditionRow(
                condition_index=k,
                condition_weight=float(weights[k]),
                condition_scores=scores,
                condition_root_value=float(cond["root_value"]),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Toy producer: uniform-determinization conditional shard
# ---------------------------------------------------------------------------


def _extend_base_arrays_with_conditions(
    arrays: dict[str, np.ndarray],
    *,
    condition_ids: tuple[str, ...],
    condition_label_format: str,
    condition_schema_digest: str,
    dataset_run_fingerprint: str | None,
) -> dict[str, np.ndarray]:
    """Add the per-decision condition axis to a flat-MC shard's arrays.

    Uniform-determinization toy: K = len(condition_ids) conditions per
    decision, every condition's `condition_scores` = the flat-MC `scores`
    (identical across conditions), uniform weights 1/K. The condition_index
    tag varies but carries no hidden information.
    """

    if SCORE_KEY not in arrays:
        raise ConditionalDistillError("base shard lacks flat-MC scores")
    scores = np.asarray(arrays[SCORE_KEY], dtype=np.float32)  # (D, max_actions)
    decisions, max_actions = scores.shape
    k = len(condition_ids)
    if k < 1:
        raise ConditionalDistillError("at least one condition is required")
    condition_count = np.full(decisions, k, dtype=np.int16)
    condition_index = np.full((decisions, k), -1, dtype=np.int16)
    condition_weight = np.zeros((decisions, k), dtype=np.float32)
    condition_scores = np.full((decisions, k, max_actions), -1.0, dtype=np.float32)
    for ki in range(k):
        condition_index[:, ki] = ki
        condition_weight[:, ki] = 1.0 / k
        condition_scores[:, ki, :] = scores
    arrays[CONDITION_COUNT_KEY] = condition_count
    arrays[CONDITION_INDEX_KEY] = condition_index
    arrays[CONDITION_WEIGHT_KEY] = condition_weight
    arrays[CONDITION_SCORES_KEY] = condition_scores

    import json as _json

    arrays["provenance"] = np.array(
        _json.dumps(
            _conditional_provenance(
                base_provenance=_json.loads(str(arrays["provenance"]))
                if "provenance" in arrays
                else {},
                condition_ids=condition_ids,
                condition_label_format=condition_label_format,
                condition_schema_digest=condition_schema_digest,
                dataset_run_fingerprint=dataset_run_fingerprint,
            )
        )
    )
    return arrays


def _conditional_provenance(
    *,
    base_provenance: Mapping[str, Any],
    condition_ids: tuple[str, ...],
    condition_label_format: str,
    condition_schema_digest: str,
    dataset_run_fingerprint: str | None,
) -> dict[str, Any]:
    out = dict(base_provenance)
    out["condition_label_format"] = condition_label_format
    out["condition_schema_digest"] = condition_schema_digest
    out["condition_ids"] = list(condition_ids)
    out["condition_count"] = len(condition_ids)
    out["policy_target_kind"] = "score_softmax"
    out["value_target_kind"] = "terminal_outcome"
    if dataset_run_fingerprint is not None:
        out["dataset_run_fingerprint"] = dataset_run_fingerprint
    return out


def generate_uniform_determinization_shard(
    *,
    num_games: int,
    sims: int,
    seed: int,
    out_path: str | Path,
    condition_ids: tuple[str, ...] = CONDITION_ROLES,
    game_offset: int = 0,
    dataset_run_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Produce a uniform-determinization conditional shard.

    Runs cheap flat-MC self-play (`distill.generate_selfplay_shard`) and
    extends the resulting shard with K = len(condition_ids) conditions per
    decision, all sharing the flat-MC scores (uniform determinization) with
    uniform weights 1/K. The shard conforms to the pinned
    `ConditionalStrategyResult` shape via the provenance
    `condition_schema_digest`; the condition_index tag varies but carries no
    hidden information, so the pre-registered strength prediction is ~0 gap.
    """

    out_path = Path(out_path)
    base_summary = generate_selfplay_shard(
        num_games=num_games,
        sims=sims,
        seed=seed,
        game_offset=game_offset,
        out_path=out_path,
        dataset_run_fingerprint=dataset_run_fingerprint,
    )
    with np.load(out_path) as data:
        arrays = {key: data[key] for key in data.files}
    digest = conditional_strategy_shape_digest()
    arrays = _extend_base_arrays_with_conditions(
        arrays,
        condition_ids=condition_ids,
        condition_label_format=CONDITION_LABEL_FORMAT,
        condition_schema_digest=digest,
        dataset_run_fingerprint=dataset_run_fingerprint,
    )
    temporary = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as destination:
            np.savez_compressed(destination, **arrays)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)
    provenance = _conditional_provenance(
        base_provenance=base_summary["provenance"],
        condition_ids=condition_ids,
        condition_label_format=CONDITION_LABEL_FORMAT,
        condition_schema_digest=digest,
        dataset_run_fingerprint=dataset_run_fingerprint,
    )
    return {
        **base_summary,
        "provenance": provenance,
        "condition_count": len(condition_ids),
        "condition_label_format": CONDITION_LABEL_FORMAT,
        "condition_schema_digest": digest,
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Loader: conditional shards -> per-(decision, condition) training rows
# ---------------------------------------------------------------------------


def _load_conditional_arrays(paths: list[str | Path]) -> list[dict[str, np.ndarray]]:
    import json as _json

    shards = [np.load(Path(p)) for p in paths]
    for shard in shards:
        for key in CONDITION_KEYS:
            if key not in shard:
                raise ConditionalDistillError(
                    f"conditional shard missing required key {key!r}"
                )
        provenance = (
            _json.loads(str(shard["provenance"])) if "provenance" in shard else {}
        )
        digest = provenance.get("condition_schema_digest")
        if digest != conditional_strategy_shape_digest():
            raise ConditionalDistillError(
                "conditional shard condition_schema_digest does not match the "
                "pinned ConditionalStrategyResult shape contract"
            )
        if provenance.get("policy_target_kind") != "score_softmax":
            raise ConditionalDistillError(
                "conditional shard policy_target_kind must be score_softmax"
            )
        if provenance.get("value_target_kind") != "terminal_outcome":
            raise ConditionalDistillError(
                "conditional shard value_target_kind must be terminal_outcome"
            )
    return shards


def load_conditional_shards(
    paths: list[str | Path],
) -> dict[str, np.ndarray]:
    """Load conditional shards and expand to per-(decision, condition) rows.

    Each decision d with K conditions becomes K rows. The public observation
    fields are repeated K times; per-row `scores` = `condition_scores[d, k, :]`
    (the per-condition strategy as score-softmax targets); per-row `action` =
    the per-condition argmax over valid actions; per-row `condition_index`/
    `condition_weight` carry the side input the conditioned `Agent` reads. The
    train/val split is by game (all K rows of a decision share a game_index).
    """

    import json as _json

    shards = _load_conditional_arrays(paths)
    expanded: dict[str, list[np.ndarray]] = {}
    round_cols: list[np.ndarray] = []
    for shard in shards:
        condition_count = np.asarray(shard[CONDITION_COUNT_KEY], dtype=np.int64)
        condition_index = np.asarray(shard[CONDITION_INDEX_KEY], dtype=np.int64)
        condition_weight = np.asarray(shard[CONDITION_WEIGHT_KEY], dtype=np.float32)
        condition_scores = np.asarray(shard[CONDITION_SCORES_KEY], dtype=np.float32)
        actions_valid = np.asarray(shard["actions_valid"], dtype=np.float32)
        decisions = len(condition_count)
        k_max = condition_index.shape[1]
        decision_rows: list[int] = []
        condition_rows: list[int] = []
        for d in range(decisions):
            k = int(condition_count[d])
            if k < 1 or k > k_max:
                raise ConditionalDistillError(
                    f"decision {d} has invalid condition_count {k}"
                )
            for ki in range(k):
                decision_rows.append(d)
                condition_rows.append(ki)
        idx = np.asarray(decision_rows, dtype=np.int64)
        cond_k = np.asarray(condition_rows, dtype=np.int64)

        # Per-row public obs + meta, repeated per condition. ``action`` and
        # ``scores`` are replaced per condition below, so skip them here to
        # avoid double-appending.
        for key in list(OBS_KEYS) + list(META_KEYS):
            if key in ("action", SCORE_KEY) or key not in shard:
                continue
            col = shard[key]
            expanded.setdefault(key, []).append(col[idx])
        # Per-row scores = per-condition strategy scores.
        per_row_scores = condition_scores[idx, cond_k]
        # Guarantee -1 padding matches the legal mask exactly.
        legal = actions_valid[idx] > 0
        per_row_scores = np.where(legal, per_row_scores, -1.0).astype(np.float32)
        expanded.setdefault(SCORE_KEY, []).append(per_row_scores)
        # Per-row action = per-condition argmax over valid actions.
        per_row_action = np.argmax(
            np.where(legal, per_row_scores, -np.inf), axis=1
        ).astype(np.int16)
        expanded.setdefault("action", []).append(per_row_action)
        # Side inputs.
        expanded.setdefault("condition_index", []).append(
            condition_index[idx, cond_k].astype(np.int64)
        )
        expanded.setdefault("condition_weight", []).append(
            condition_weight[idx, cond_k].astype(np.float32)
        )
        provenance = (
            _json.loads(str(shard["provenance"])) if "provenance" in shard else {}
        )
        round_cols.append(
            np.full(len(idx), int(provenance.get("round", -1)), dtype=np.int16)
        )
    out = {key: np.concatenate(parts) for key, parts in expanded.items()}
    out["round"] = np.concatenate(round_cols)
    # Game-index offsetting across rounds mirrors distill.load_shards.
    unique_rounds = np.unique(out["round"])
    if len(unique_rounds) > 1:
        game_index = out["game_index"].astype(np.int64)
        offset = 0
        for r in unique_rounds:
            rows = out["round"] == r
            game_index[rows] += offset
            offset = int(game_index[rows].max()) + 1
        out["game_index"] = game_index
    return out


def with_neutral_condition(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a copy of an expanded conditional dataset with a neutral condition.

    The condition side input is replaced by the neutral True/uniform condition
    (`condition_index=0, condition_weight=1.0`) on every row. Used for the
    unconditioned matched-control arms: same rows, same targets, only the
    condition feature content is masked to the uninformative prior.
    """

    out = dict(dataset)
    out["condition_index"] = np.zeros(len(dataset["action"]), dtype=np.int64)
    out["condition_weight"] = np.ones(len(dataset["action"]), dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# Dataset directory builder: shards + sidecars + manifest (schema_version 2)
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    import hashlib as _hashlib

    h = _hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    import json as _json
    import os as _os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{_os.getpid()}.tmp")
    with tmp.open("w") as fh:
        _json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        _os.fsync(fh.fileno())
    tmp.replace(path)


def build_conditional_dataset(
    out_dir: str | Path,
    *,
    num_games: int,
    games_per_shard: int,
    sims: int,
    seed: int,
    condition_ids: tuple[str, ...] = CONDITION_ROLES,
) -> dict[str, Any]:
    """Generate a conditional dataset directory: shards + sidecars + manifest.

    The directory layout matches what
    ``run_belief_conditioned_snapshot.freeze_conditional_snapshot`` expects:
    one ``shard_XXXXX.npz`` + ``shard_XXXXX.json`` sidecar per shard, and a
    ``manifest.json`` with ``schema_version: 2``, a ``run_contract``, a
    ``run_fingerprint`` (SHA-256 over the contract), and the shard summaries.

    Each shard is produced by ``generate_uniform_determinization_shard``
    (uniform-determinization toy: K conditions per decision, all sharing the
    flat-MC scores, uniform weights 1/K). The condition_index tag varies but
    carries no hidden information, so the pre-registered strength prediction
    is ~0 gap. This is plumbing/measurement evidence, not a strength claim.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_contract: dict[str, Any] = {
        "schema_version": 1,
        "games": num_games,
        "games_per_shard": games_per_shard,
        "sims": sims,
        "seed": seed,
        "condition_count": len(condition_ids),
        "condition_label_format": CONDITION_LABEL_FORMAT,
        "condition_ids": list(condition_ids),
        "teacher_spec": {"kind": "flat_mc", "sims": sims},
        "source_commit": _git_commit(),
    }
    run_fingerprint = _canonical_json_sha256(run_contract)
    shard_count = max(1, num_games // games_per_shard)
    summaries: list[dict[str, Any]] = []
    for index in range(shard_count):
        shard_path = out_dir / f"shard_{index:05d}.npz"
        game_offset = index * games_per_shard
        shard_games = min(games_per_shard, num_games - game_offset)
        summary = generate_uniform_determinization_shard(
            num_games=shard_games,
            sims=sims,
            seed=seed + index * 1_000_000,
            out_path=shard_path,
            condition_ids=condition_ids,
            game_offset=game_offset,
            dataset_run_fingerprint=run_fingerprint,
        )
        summary["shard_index"] = index
        summary["run_fingerprint"] = run_fingerprint
        summary["sha256"] = _file_sha256(shard_path)
        summary["out_path"] = str(shard_path)
        _atomic_json(shard_path.with_suffix(".json"), summary)
        summaries.append(summary)
    manifest = {
        "schema_version": 2,
        "status": "completed",
        "games": num_games,
        "games_per_shard": games_per_shard,
        "run_fingerprint": run_fingerprint,
        "run_contract": run_contract,
        "shards": sorted(summaries, key=lambda item: item["shard_index"]),
    }
    _atomic_json(out_dir / "manifest.json", manifest)
    return manifest


__all__ = [
    "CONDITION_COUNT_KEY",
    "CONDITION_INDEX_KEY",
    "CONDITION_WEIGHT_KEY",
    "CONDITION_SCORES_KEY",
    "CONDITION_ROOT_VALUE_KEY",
    "CONDITION_KEYS",
    "CONDITION_ROW_KEYS",
    "CONDITION_ROLES",
    "CONDITION_LABEL_FORMAT",
    "DEFAULT_CONDITION_COUNT",
    "CONDITIONAL_STRATEGY_SHAPE",
    "ConditionalDistillError",
    "conditional_strategy_shape_digest",
    "assert_conforms_to_conditional_strategy_shape",
    "ConditionRow",
    "viewer_safe_to_condition_rows",
    "generate_uniform_determinization_shard",
    "load_conditional_shards",
    "with_neutral_condition",
    "build_conditional_dataset",
]
