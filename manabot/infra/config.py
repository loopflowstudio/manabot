"""
Configuration models shared across training, simulation and runtime setup.
"""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from manabot.env.legacy_input import warn_deprecated_input
from managym import (
    WORLD_SETUP,
    Deck,
    DeckImport,
    Decklist,
    ManaFloats,
    ManaPayment,
    PlayerConfig,
)
from managym.disclosure import DecklistDisclosure

# The policy input a fresh configuration selects. Input v11 gives each game
# thing one object row (cards outside the battlefield and stack, permanents
# with their characteristics, a stack segment) and carries v10's turn state.
# Inputs v9 and v10 (v9 plus turn state) are deprecated and read only by
# checkpoints trained on them (`manabot.env.legacy_input`, removed in 0.3.0);
# earlier inputs run from the revision that recorded them.
PolicyInputVersion = Literal[9, 10, 11]
POLICY_INPUT_VERSION: PolicyInputVersion = 11
SUPPORTED_POLICY_INPUTS: tuple[PolicyInputVersion, ...] = (9, 10, 11)
DEPRECATED_POLICY_INPUTS: tuple[PolicyInputVersion, ...] = (9, 10)
# What a saved regime's observation settings mean when they omit the version:
# the fresh default when they were written.
SAVED_POLICY_INPUT_VERSION: PolicyInputVersion = 9
# The last revision whose encoders, bindings and model accept inputs v1-v8.
LAST_MULTI_INPUT_REVISION = "ac601834d"
# How the critic summarizes the final object board; see ActorCritic._value_from_objects.
ValueAggregation = Literal[
    "historical_mean", "masked_mean", "value_token", "attention_readout"
]


def _default_deck() -> dict[str, int]:
    return {
        "Mountain": 12,
        "Forest": 12,
        "Llanowar Elves": 18,
        "Gray Ogre": 18,
    }


def _default_runs_dir() -> Path:
    runs_dir = os.getenv("MANABOT_RUNS_DIR")
    if runs_dir:
        return Path(runs_dir)
    return Path.cwd() / ".runs"


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UnsupportedPolicyInput(ValueError):
    """A checkpoint, replay or saved configuration bound to another policy input.

    `version` is the saved version; a saved configuration that omits it was
    written for policy input v1.
    """

    def __init__(self, version: object) -> None:
        self.version = version
        super().__init__(
            f"the checkpoint was written for policy input v{version} and must "
            "be run from the revision that recorded it; this revision supports "
            f"policy inputs {' and '.join(f'v{v}' for v in SUPPORTED_POLICY_INPUTS)}, "
            f"and {LAST_MULTI_INPUT_REVISION} is the last revision supporting v1-v8"
        )


class ObservationConfig(BaseConfig):
    # Always serialized, so a saved configuration names its input.
    policy_input_version: PolicyInputVersion = POLICY_INPUT_VERSION
    # Local fact rows; shared effects have their own fixed buffers.
    max_facts: int = Field(default=512, gt=0)

    max_cards_per_player: int = 60
    # 30 -> 40: token-heavy GW Allies games exceed 30 battlefield entries.
    max_permanents_per_player: int = 40
    # Input v11 stack rows; overflow is a capacity error.
    max_stack_objects: int = Field(
        default=32, gt=0, exclude_if=lambda value: value == 32
    )
    # Naming is bounded group prompts, so this capacity is independent of the
    # name vocabulary. A pilot may explicitly retain a legal prefix.
    max_actions: int = Field(default=512, gt=0)
    action_overflow: Literal["error", "first-legal"] = Field(
        default="error", exclude_if=lambda value: value == "error"
    )
    mana_payment: ManaPayment = Field(
        default="manual", exclude_if=lambda value: value == "manual"
    )
    mana_floats: ManaFloats = Field(
        default="off", exclude_if=lambda value: value == "off"
    )
    max_focus_objects: int = Field(default=2, gt=0)
    # Memory capacities are explicit until the replay census sizes the pilot.
    decision_memory_include_own: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    max_decision_records: int = Field(
        default=0, ge=0, exclude_if=lambda value: value == 0
    )
    max_decision_reference_details: int = Field(
        default=0, ge=0, exclude_if=lambda value: value == 0
    )
    max_decision_references: int = Field(
        default=0, ge=0, exclude_if=lambda value: value == 0
    )
    max_decision_members: int = Field(
        default=0, ge=0, exclude_if=lambda value: value == 0
    )
    max_events: int = 32
    policy_history_version: Literal[0, 1] = Field(
        default=0, exclude_if=lambda value: value == 0
    )

    @field_validator("policy_input_version", mode="before")
    @classmethod
    def reject_other_inputs(cls, value: object) -> object:
        """Name the recording revision instead of a bare literal mismatch."""
        if value not in SUPPORTED_POLICY_INPUTS:
            raise UnsupportedPolicyInput(value)
        if value in DEPRECATED_POLICY_INPUTS:
            warn_deprecated_input(value)
        return value

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> "ObservationConfig":
        """Read a saved configuration, rejecting any other policy input.

        A missing version means the input-v1 contract, never today's default.
        Raises `UnsupportedPolicyInput` before any other field is validated.
        """
        version = payload.get("policy_input_version", 1)
        if version not in SUPPORTED_POLICY_INPUTS:
            raise UnsupportedPolicyInput(version)
        return cls(**payload)


class MatchConfig(BaseConfig):
    """Parameters passed to the match builder."""

    # `WORLD_SETUP` submits the decks to the installed world's admission;
    # `None` runs them as given. Every match runs on the world's one catalog.
    # Frozen regimes name this selector `content_pack` and may record an
    # authored setup's `pack_key` here; that label is kept so their identities
    # hold, and it runs like `None`.
    world: str | None = Field(
        default=None,
        validation_alias=AliasChoices("world", "content_pack"),
        exclude_if=lambda value: value is None,
    )
    # Private import provenance: each seat's admitted decklist, retained beside
    # (never instead of) that seat's deck. Viewer frames never carry it.
    hero_source: Decklist | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    villain_source: Decklist | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    hero: str = "gaea"
    villain: str = "urza"
    hero_deck: dict[str, int] = Field(default_factory=_default_deck)
    villain_deck: dict[str, int] = Field(default_factory=_default_deck)
    hero_sideboard: dict[str, int] = Field(default_factory=dict)
    villain_sideboard: dict[str, int] = Field(default_factory=dict)
    # Open decklists disclose each player's main deck and sideboard to the
    # opponent; closed disclose neither. One setting covers both players.
    # Omitted when open, so earlier configurations keep their identity.
    decklists: DecklistDisclosure = Field(
        default="open", exclude_if=lambda value: value == "open"
    )

    mana_payment: tuple[ManaPayment, ManaPayment] = Field(
        default=("manual", "manual"),
        exclude_if=lambda value: value == ("manual", "manual"),
    )
    mana_floats: tuple[ManaFloats, ManaFloats] = Field(
        default=("off", "off"), exclude_if=lambda value: value == ("off", "off")
    )

    @field_validator("world")
    @classmethod
    def validate_world(cls, value: str | None) -> str | None:
        if value is not None and value != WORLD_SETUP:
            import managym

            managym.authored_deck_keys(value)  # raises for an unknown setup
        return value

    @model_validator(mode="after")
    def validate_sources(self) -> "MatchConfig":
        """Each retained source must re-admit to its seat's deck in the installed world."""
        for source, main, sideboard in (
            (self.hero_source, self.hero_deck, self.hero_sideboard),
            (self.villain_source, self.villain_deck, self.villain_sideboard),
        ):
            if source is None:
                continue
            if self.world != WORLD_SETUP:
                raise ValueError("an imported decklist plays in the world")
            source.validate_deck(Deck(main, sideboard))
        return self

    @classmethod
    def from_players(
        cls,
        first: PlayerConfig,
        second: PlayerConfig,
        *,
        decklists: DecklistDisclosure,
        mana_floats: tuple[ManaFloats, ManaFloats] = ("off", "off"),
        mana_payment: tuple[ManaPayment, ManaPayment] = ("manual", "manual"),
    ) -> "MatchConfig":
        """Record a native root in seat order, rejecting asymmetric match settings."""
        if first.world != second.world:
            raise ValueError(
                f"players require the same world: seat 0 ({first.name}) has {first.world!r}, "
                f"seat 1 ({second.name}) has {second.world!r}"
            )
        return cls(
            hero=first.name,
            villain=second.name,
            hero_deck=first.decklist,
            villain_deck=second.decklist,
            hero_sideboard=first.sideboard,
            villain_sideboard=second.sideboard,
            world=first.world,
            decklists=decklists,
            mana_floats=mana_floats,
            mana_payment=mana_payment,
        )

    @classmethod
    def from_import(cls, hero: DeckImport, villain: DeckImport) -> "MatchConfig":
        """The playable setup of two admitted decklists, in the world."""
        return cls(
            world=WORLD_SETUP,
            hero_source=hero.decklist,
            villain_source=villain.decklist,
            hero_deck=hero.deck.main,
            villain_deck=villain.deck.main,
            hero_sideboard=hero.deck.sideboard,
            villain_sideboard=villain.deck.sideboard,
        )

    @field_validator("hero_sideboard", "villain_sideboard", mode="before")
    @classmethod
    def validate_sideboard_counts(cls, value: Any) -> Any:
        if not isinstance(value, dict) or any(
            not isinstance(name, str)
            or not name
            or type(count) is not int
            or count <= 0
            for name, count in value.items()
        ):
            raise ValueError("sideboard must map card names to positive integer counts")
        return value

    @classmethod
    def authored(
        cls,
        setup: str,
        hero_deck: str,
        villain_deck: str,
        *,
        hero: str = "gaea",
        villain: str = "urza",
        decklists: DecklistDisclosure = "open",
    ) -> "MatchConfig":
        """Resolve both complete setups of an authored setup (preset or `pack_key`)."""
        import managym

        first = managym.authored_deck_setup(setup, hero_deck)
        second = managym.authored_deck_setup(setup, villain_deck)
        return cls(
            hero=hero,
            villain=villain,
            hero_deck=first.decklist,
            villain_deck=second.decklist,
            hero_sideboard=first.sideboard,
            villain_sideboard=second.sideboard,
            decklists=decklists,
        )


class RuntimeConfig(BaseConfig):
    """Configuration for experiment tracking and runtime setup."""

    name: str = "manabot"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = "cpu"
    runs_dir: Path = Field(default_factory=_default_runs_dir)
    log_level: str = "INFO"


# Compound-decision policies (a learned ragged decoder over native compound
# offers) were removed; their checkpoints and regimes run only from the source
# that trained them.
COMPOUND_DECISIONS_REMOVED = (
    "compound-decision policies are no longer executable; reproduce them at "
    "commit ba3cd58de25bcc5d0b32e0fa3c546ee3d27be52b"
)
# The pooled token stream (`token_stream="pooled"`) was removed the same way.
POOLED_TOKEN_STREAM_REMOVED = (
    "token_stream='pooled' was removed after its pilot closed negative "
    "(experiments/reports/token_stream.md); live-token attention replaces it. "
    "Reproduce it at commit 439379869541cf29e781d7215b088e828041889e"
)


# The narrowest `ModelConfig.hidden_dim` whose MLX forwards default to
# bfloat16 (`manabot.backends.mlx.arrays.default_precision`); torch settings
# state their precision. BF16 learner updates were
# first faster than float32 at this width on an L4 with bucketed shapes
# (experiments/reports/cuda_precision.md); learning evidence at w384l8 is in
# experiments/reports/bf16_behavior_logs.md. Narrower models and CPU stay float32.
BF16_MIN_HIDDEN_DIM = 256


class ModelConfig(BaseConfig):
    """Model construction contract, independent of learning and execution settings.

    Checkpoints retain the serialized ``agent_hypers`` dictionary key; renaming
    this Python type changes neither field meanings nor ordinary reload admission.
    """

    decision_memory: Literal["none", "transformer", "gru"] = Field(
        default="none", exclude_if=lambda value: value == "none"
    )
    decision_memory_summaries: Literal[1, 4] = Field(
        default=1, exclude_if=lambda value: value == 1
    )
    decision_memory_gate_scale: float = Field(
        default=1.0, gt=0, allow_inf_nan=False, exclude_if=lambda value: value == 1.0
    )
    # Public identity/history v1; false preserves historical receipt bytes.
    recent_events: bool = Field(default=False, exclude_if=lambda value: not value)
    # The semantic vocabulary the model reads: the world's one catalog. Fresh
    # recipes write "world"; older labels in `FROZEN_VOCABULARY_LABELS` are
    # accepted on read and bind the same catalog. Frozen checkpoints and
    # recipes name the field `semantic_pack`.
    world: str | None = Field(
        default=None, validation_alias=AliasChoices("world", "semantic_pack")
    )
    # Serialized architecture choice; categorical logits are loss/draw/win.
    value_kind: Literal["scalar", "categorical_wdl"] = "scalar"
    value_aggregation: ValueAggregation = "historical_mean"
    # Scale each action embedding by the pooled game state before the policy
    # head (FiLM). Omitted is the historical additive policy, so frozen
    # recipes and checkpoints keep their meaning.
    policy_film: bool = Field(default=False, exclude_if=lambda value: not value)
    # Scale each offer's own features by its focused post-attention objects
    # before the action layer, so offers naming one object differ by a
    # function of it from initialization. Omitted is the historical head.
    focus_film: bool = Field(default=False, exclude_if=lambda value: not value)
    # One normalized token stream: a LayerNorm on every stream summed into an
    # object token (numeric projections, definitions, the knowledge pool,
    # shared context), pre-normalization attention and a final norm. Omitted
    # is the historical post-normalization layout with `/ 60` knowledge pools.
    stream_norm: bool = Field(default=False, exclude_if=lambda value: not value)
    attention_layers: int = Field(default=1, ge=1)
    # Omitted defaults preserve historical recipe and checkpoint identities.
    attention_feedforward_dim: int | None = Field(
        default=None, ge=1, exclude_if=lambda value: value is None
    )
    # How fact and shared-effect ids embed: `catalog` reads the card encoder's
    # exact token rows and definition rows; `hashed` learns 4096 hashed rows
    # per id column. Models without a world have only the hashed layout. Saved
    # models with world="world" that omit the field use hashed rows (`from_checkpoint`);
    # regimes omit the fresh default.
    record_vocabulary: Literal["hashed", "catalog"] = Field(
        default="catalog", exclude_if=lambda value: value == "catalog"
    )
    # Whether each public name vector is added onto its definition's program,
    # so every card token carries a name stream beside its rules program.
    # Fresh models keep names on NameCard candidates only; checkpoints that
    # omit the field predate the choice and add them (`from_checkpoint`).
    definition_names: bool = Field(default=False, exclude_if=lambda value: not value)
    # Shared embedding space for game objects and actions.
    hidden_dim: int = 64
    # Number of attention heads used in the GameObjectAttention layer.
    num_attention_heads: int = 4
    attention_on: bool = True
    # Canonical belief rows are an opt-in checkpoint ABI. A positive bucket
    # count requires explicit belief tensors on every forward pass; there is
    # no neutral or positional-condition fallback on this path.
    belief_count_buckets: int = 0
    belief_card_vocab_size: int = 0
    belief_owner_role_vocab_size: int = 2
    belief_hidden_zone_vocab_size: int = 7

    @model_validator(mode="before")
    @classmethod
    def reject_compound_decisions(cls, data: Any) -> Any:
        """Admit the retired flag only when false; never load it as a flat policy."""
        if not isinstance(data, dict) or "compound_decisions" not in data:
            return data
        data = dict(data)
        if data.pop("compound_decisions") is not False:
            raise ValueError(COMPOUND_DECISIONS_REMOVED)
        return data

    @model_validator(mode="before")
    @classmethod
    def reject_pooled_token_stream(cls, data: Any) -> Any:
        """Admit the retired `token_stream` only as `tables`, the one remaining
        attention layout; a saved `pooled` stream is refused by name."""
        if not isinstance(data, dict) or "token_stream" not in data:
            return data
        data = dict(data)
        if data.pop("token_stream") != "tables":
            raise ValueError(POOLED_TOKEN_STREAM_REMOVED)
        return data

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> "ModelConfig":
        """Read a saved model configuration.

        A checkpoint that omits the record vocabulary or definition names
        predates the choice, so it means the frozen hashed ids and names added
        onto definitions where the choice exists (with a world). Without a
        world both values build the same model, and omission reads as the
        fresh default so it equals its regime's configuration.
        """
        config = cls(**payload)
        if config.world is None:
            return config
        predated = {
            key: value
            for key, value in (
                ("record_vocabulary", "hashed"),
                ("definition_names", True),
            )
            if key not in payload
        }
        return cls(**{**payload, **predated}) if predated else config

    def checkpoint_payload(self) -> dict[str, Any]:
        """The serialized configuration with the record vocabulary and definition
        names explicit, so `from_checkpoint` reads a fresh model back as fresh."""
        return {
            **self.model_dump(),
            "record_vocabulary": self.record_vocabulary,
            "definition_names": self.definition_names,
        }

    def catalog_records(self) -> bool:
        """Whether fact and shared-effect ids read the card vocabulary."""
        return self.record_vocabulary == "catalog" and self.world is not None

    @model_validator(mode="after")
    def validate_value_architecture(self) -> "ModelConfig":
        if self.decision_memory != "none" and (
            not self.attention_on or not self.policy_film or self.recent_events
        ):
            raise ValueError(
                "decision memory requires attention and policy_film, with recent_events disabled"
            )
        if self.hidden_dim < 1 or self.num_attention_heads < 1:
            raise ValueError("embedding width and head count must be positive")
        if self.attention_on and self.hidden_dim % self.num_attention_heads:
            raise ValueError("embedding width must be divisible by attention heads")
        if self.stream_norm and not self.attention_on:
            raise ValueError("stream_norm requires attention_on")
        if not self.attention_on and (
            self.value_aggregation == "value_token"
            or self.attention_layers != 1
            or self.attention_feedforward_dim is not None
        ):
            raise ValueError("value token and stacked layers require attention")
        return self


class RewardConfig(BaseConfig):
    trivial: bool = False
    managym: bool = False
    win_reward: float = 1.0
    lose_reward: float = -1.0
    land_play_reward: float = 0.0
    creature_play_reward: float = 0.0
    opponent_life_loss_reward: float = 0.0
    # Potential-based shaping (Ng, Harada & Russell 1999): adds
    # gamma * Phi(s') - Phi(s) to every hero step reward, with Phi(terminal)
    # treated as 0 so the potential telescopes out over an episode. Phi is a
    # hero-perspective board-state potential:
    #   Phi(s) = potential_land_weight * (hero bf lands - villain bf lands)
    #          + potential_creature_weight * (hero bf creatures - villain bf creatures)
    #          + potential_life_weight * (hero life - villain life) / 20
    # potential_gamma must match the training discount (train.gamma) for
    # policy invariance to hold.
    potential_enabled: bool = False
    potential_gamma: float = 0.99
    potential_land_weight: float = 0.03
    potential_creature_weight: float = 0.06
    potential_life_weight: float = 0.2


class SimulationConfig(BaseConfig):
    """Hyperparameters for model simulation."""

    hero: str = "random"
    villain: str = "default"
    num_games: int = 100
    num_threads: int = 4
    max_steps: int = 2000
    match: MatchConfig = Field(default_factory=MatchConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
