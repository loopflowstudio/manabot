"""Frozen protocol for the bounded training-regime workflow studies."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.runners.history_input import (
    CALIBRATION_SECONDS,
    TOTAL_SECONDS,
    InputBinding,
)


class EvaluationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[2] = 2
    study: Literal[
        "learning-speed",
        "ataraxos-ablations",
        "omitted-controls",
        "compound-decisions",
        "training-calibration",
        "capacity-calibration",
        "model-capacity",
        "value-models",
        "value-token-screen",
        "pooling-filter",
        "history-input",
        "depth-screen",
    ]
    purpose: Literal["workflow-smoke", "calibration", "scientific", "screening"] = (
        "workflow-smoke"
    )
    evaluation_variants: tuple[Literal["raw", "ema"], ...] = ("raw",)
    regime_digests: tuple[str, ...]
    training_seeds: tuple[int, ...] = (197,)
    paired_deals: tuple[int, ...] = (910001,)
    anchor_deals: tuple[int, ...] = (920001,)
    checkpoint_count: int = Field(default=2, ge=2)
    cost_cutoffs_seconds: tuple[float, ...] = ()
    early_progress_seconds: float | None = Field(
        default=None, gt=0, exclude_if=lambda v: v is None
    )
    progress_score: float | None = Field(
        default=None, ge=0, le=1, exclude_if=lambda v: v is None
    )
    anchors: tuple[Literal["random", "scripted-greedy", "puct-64"], ...] = ("random",)
    endpoint_paired_deals: tuple[int, ...] = ()
    endpoint_anchor_deals: tuple[int, ...] = ()
    endpoint_seed_pairs: tuple[tuple[int, int], ...] = ()
    seat_deck_legs: Literal[4] = 4
    inference: Literal["stochastic-policy-cpu-one-thread"] = (
        "stochastic-policy-cpu-one-thread"
    )
    game_seconds: float = Field(default=120, gt=0, le=120)
    max_commands: int = Field(default=10000, gt=0, le=10000)
    process_seconds: float = Field(default=900, gt=0, le=168 * 3600)
    selection: Literal[
        "all-completed-cutoffs-raw", "all-completed-cutoffs-raw-and-ema"
    ] = "all-completed-cutoffs-raw"
    uncertainty: Literal["cross-seed-unavailable", "paired-seed-descriptive"] = (
        "cross-seed-unavailable"
    )
    incomplete: Literal["fail-retain-all-attempts"] = "fail-retain-all-attempts"

    @model_validator(mode="after")
    def disjoint(self) -> "EvaluationProtocol":
        if (
            self.study
            in {"value-token-screen", "pooling-filter", "history-input", "depth-screen"}
        ) != (self.purpose == "screening"):
            raise ValueError(
                "value-token-screen requires its distinct screening purpose"
            )
        if self.purpose == "screening" and (
            len(self.training_seeds) != 3
            or self.anchors != ("scripted-greedy",)
            or self.checkpoint_count != 2
            or len(self.anchor_deals) != 25
            or self.paired_deals
            or self.endpoint_paired_deals
            or self.endpoint_anchor_deals
            or self.endpoint_seed_pairs
            or self.cost_cutoffs_seconds
            or (
                not TOTAL_SECONDS - CALIBRATION_SECONDS
                < self.process_seconds
                <= TOTAL_SECONDS
                if self.study == "history-input"
                else self.process_seconds != 28800
            )
            or self.uncertainty != "paired-seed-descriptive"
        ):
            raise ValueError(
                "screen requires three seeds, two checkpoints, 25 four-leg scripted deals and eight hours"
            )
        if self.study == "model-capacity" and (
            self.early_progress_seconds is None or self.progress_score is None
        ):
            raise ValueError("capacity requires an early window and threshold")
        expected_selection = (
            "all-completed-cutoffs-raw-and-ema"
            if "ema" in self.evaluation_variants
            else "all-completed-cutoffs-raw"
        )
        if self.selection != expected_selection:
            raise ValueError("selection label must agree with evaluation variants")
        if not self.evaluation_variants or len(set(self.evaluation_variants)) != len(
            self.evaluation_variants
        ):
            raise ValueError("evaluation variants must be nonempty and unique")
        if self.study != "omitted-controls" and self.evaluation_variants != ("raw",):
            raise ValueError("existing frozen studies evaluate raw only")
        if (
            any(c <= 0 for c in self.cost_cutoffs_seconds)
            or tuple(sorted(set(self.cost_cutoffs_seconds)))
            != self.cost_cutoffs_seconds
        ):
            raise ValueError("cost cutoffs must be positive and strictly increasing")
        if self.purpose == "scientific" and not self.cost_cutoffs_seconds:
            raise ValueError("scientific profiles require declared cost cutoffs")
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("training seeds must be nonempty and unique")
        if any(seed < 0 or seed >= 100000 for seed in self.training_seeds):
            raise ValueError("training seeds must be in [0, 100000)")
        if self.purpose == "workflow-smoke" and (
            self.process_seconds > 900 or len(self.training_seeds) != 1
        ):
            raise ValueError("smoke is limited to 900 seconds and one seed")
        if self.purpose == "scientific" and (
            len(self.training_seeds) < 3
            or self.uncertainty != "paired-seed-descriptive"
        ):
            raise ValueError(
                "scientific profiles require at least three seeds and declared uncertainty"
            )
        if not self.anchors or len(set(self.anchors)) != len(self.anchors):
            raise ValueError("anchors must be nonempty and unique")
        if self.purpose == "scientific":
            if not self.endpoint_paired_deals or not self.endpoint_anchor_deals:
                raise ValueError("scientific profiles require untouched endpoint deals")
            if set(self.anchors) != {"random", "scripted-greedy", "puct-64"}:
                raise ValueError(
                    "scientific profiles require all three fixed baselines"
                )
            expected_pairs = (
                {(a, b) for a in self.training_seeds for b in self.training_seeds}
                if self.study == "learning-speed"
                else {(s, s) for s in self.training_seeds}
            )
            if set(self.endpoint_seed_pairs) != expected_pairs or len(
                self.endpoint_seed_pairs
            ) != len(expected_pairs):
                raise ValueError(
                    "endpoint seed schedule must cover the declared cross-seed comparison"
                )
        families = (
            self.paired_deals,
            self.anchor_deals,
            self.endpoint_paired_deals,
            self.endpoint_anchor_deals,
        )
        flat = [s for family in families for s in family]
        if len(flat) != len(set(flat)):
            raise ValueError(
                "development and endpoint deal families must be disjoint and unique"
            )
        if any(s < 900000 for s in flat):
            raise ValueError("evaluation deals must use reserved family >=900000")
        expected_counts = {
            "learning-speed": {2},
            "ataraxos-ablations": {5},
            "compound-decisions": {4},
            "omitted-controls": {1, 2},
            "training-calibration": {1},
            "capacity-calibration": {3},
            "model-capacity": {3},
            "value-models": {8},
            "value-token-screen": {3},
            "pooling-filter": {4},
            "history-input": {2},
            "depth-screen": {2},
        }[self.study]
        if len(self.regime_digests) not in expected_counts or any(
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            for digest in self.regime_digests
        ):
            raise ValueError("protocol must bind every resolved recipe digest")
        if (
            not self.paired_deals and self.purpose != "screening"
        ) or not self.anchor_deals:
            raise ValueError("evaluation deal families must be nonempty")
        return self


class ResolvedStudy(BaseModel):
    """Explicit recipes and allocation; the runner never scales smoke implicitly."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: EvaluationProtocol
    recipes: tuple[dict, ...]
    allocation_seconds: float = Field(gt=0, le=168 * 3600)
    prior_campaign_seconds: float = Field(ge=0)
    runtime_identities: dict[str, str] = Field(default_factory=dict)
    input_bindings: tuple[InputBinding, ...] = Field(
        default=(), exclude_if=lambda v: not v
    )
    projected_disk_bytes: int = Field(default=0, ge=0)
    disk_reserve_bytes: int = Field(default=4 * 1024**3, ge=0)
    calibration_evidence: str

    @model_validator(mode="after")
    def allocation(self) -> "ResolvedStudy":
        from manabot.arena.models import canonical_sha256
        from manabot.training.models import TrainingRegime, TrainSelfPlay

        recipes = [TrainingRegime.model_validate(r) for r in self.recipes]
        if (
            tuple(canonical_sha256(r.model_dump(mode="json")) for r in recipes)
            != self.protocol.regime_digests
        ):
            raise ValueError("resolved plan recipe digests do not match")
        if self.protocol.purpose == "screening":
            from experiments.runners.value_screen import validate_screen_recipes

            if self.protocol.study in {"history-input", "depth-screen"}:
                from experiments.runners.screen_spec import specification

                spec = specification(self.protocol.study)
                SEEDS, DEALS = spec.SEEDS, spec.DEALS
                validate_plan_recipes = spec.validate_plan_recipes

                receipt = validate_plan_recipes(
                    recipes,
                    self.calibration_evidence,
                    self.runtime_identities,
                    self.input_bindings,
                )
                if (
                    self.protocol.evaluation_variants != ("raw",)
                    or self.protocol.training_seeds != SEEDS
                    or self.protocol.anchor_deals != DEALS
                    or self.protocol.game_seconds != 120
                    or self.protocol.max_commands != 10000
                    or self.protocol.early_progress_seconds is not None
                    or self.protocol.progress_score is not None
                ):
                    raise ValueError("history requires its frozen paired seeds/deals")
                if (
                    self.projected_disk_bytes != receipt.projected_disk_bytes
                    or self.disk_reserve_bytes != 4 * 1024**3
                ):
                    raise ValueError(
                        "history disk projection/reserve differs from admission"
                    )
            elif self.protocol.study == "pooling-filter":
                from experiments.runners.pooling_filter import SEEDS, validate_followup

                validate_followup(
                    recipes, self.calibration_evidence, self.runtime_identities
                )
                if (
                    self.protocol.training_seeds != SEEDS
                    or self.protocol.anchor_deals != tuple(range(961160, 961185))
                ):
                    raise ValueError(
                        "follow-up requires its fresh seeds and fixed paired deals"
                    )
            else:
                validate_screen_recipes(recipes)
            required = {
                "engine_extension_sha256",
                "engine_source_sha256",
                "content_manifest_sha256",
                "observation_abi_sha256",
                "action_abi_sha256",
                "matchup_sha256",
                "training_source_sha256",
                "study_source_sha256",
            }
            if self.protocol.study in {"history-input", "depth-screen"}:
                required.remove("observation_abi_sha256")
                required.add("runtime_environment_sha256")
            if set(self.runtime_identities) != required or any(
                len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                for v in self.runtime_identities.values()
            ):
                raise ValueError("screen must bind all runtime and study sources")
            allocation = (
                TOTAL_SECONDS if self.protocol.study == "history-input" else 28800
            )
            if self.protocol.study == "history-input":
                if (
                    self.allocation_seconds + self.prior_campaign_seconds != allocation
                    or self.prior_campaign_seconds >= CALIBRATION_SECONDS
                    or receipt.seconds < self.prior_campaign_seconds
                    or self.protocol.process_seconds != self.allocation_seconds
                ):
                    raise ValueError(
                        "history continuation must retain the original six-hour allocation"
                    )
            elif (
                self.allocation_seconds != allocation
                or self.prior_campaign_seconds != 0
            ):
                raise ValueError("screen owns a separate eight-hour allocation")
        if (
            self.protocol.study not in {"history-input", "depth-screen"}
            and self.input_bindings
        ):
            raise ValueError("per-arm input bindings are scoped to history-input")
        if "ema" in self.protocol.evaluation_variants and any(
            not isinstance(s, TrainSelfPlay) or s.learning.ema is None
            for r in recipes
            for s in r.stages
        ):
            raise ValueError("EMA evaluation requires EMA at every checkpoint")
        if len({r.id for r in recipes}) != len(recipes):
            raise ValueError("recipe IDs must be unique")
        if any(
            sum(s.operation != "collect_search" for s in r.stages)
            != self.protocol.checkpoint_count
            for r in recipes
        ):
            raise ValueError("every recipe must export the declared checkpoint count")
        if self.protocol.process_seconds > self.allocation_seconds:
            raise ValueError("process deadline exceeds study allocation")
        if self.prior_campaign_seconds + self.allocation_seconds > 168 * 3600:
            raise ValueError(
                "campaign exceeds 168 laptop hours including prior attempts"
            )
        if (
            sum(r.wall_seconds for r in recipes) * len(self.protocol.training_seeds)
            >= self.allocation_seconds
        ):
            raise ValueError("allocation must leave time for evaluation and reports")
        if (
            self.protocol.purpose == "scientific"
            and not self.calibration_evidence.strip()
        ):
            raise ValueError("scientific execution requires calibration evidence")
        return self
