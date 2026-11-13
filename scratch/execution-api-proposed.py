# DESIGN SKETCH ONLY — not executable. See execution-api-review.html.

# PROPOSAL — imports and signatures are illustrative, not a runnable migration.
# Existing PROTOCOL, eval(state), play(match), report() stay in this file.
# Historical frozen regimes remain evidence; these are prospective run settings.

from etude import training, track, Step
from pydantic import BaseModel, Field
from typing import Literal


class FrontierConfig(BaseModel):
    size: Literal["small", "large"]
    seed: int = 21501
    updates: int = Field(default=10_000, ge=1)
    out: Path | None = None


def train(config: FrontierConfig) -> Path:
    size, seed, updates, out = config.size, config.seed, config.updates, config.out
    large = size == "large"
    model = ModelConfig(
        world="world",
        hidden_dim=160 if large else 64,
        attention_layers=8 if large else 2,
        num_attention_heads=8 if large else 4,
    )
    learning = DampedPolicyLearning(
        iteration_schedule=IterationSchedule(
            steps=999 if large else 2474, reference_steps=10_000,
        ),
        ema=0.999,
    )
    observation = ObservationConfig(
        policy_input_version=9, max_cards_per_player=128,
        max_permanents_per_player=96, max_facts=512,
        max_actions=64, max_focus_objects=2, max_events=32,
        action_overflow="first-legal",
    )
    # Domain data loader: verifies the corpus digest and admits its decks.
    decks = load_frontier_decks(PROTOCOL.corpus_sha256)
    errors = GameErrorPolicy(
        threshold=0.25, window=100, warmup=100,
        max_game_actions=40_000, recover_capacity=True,
    )
    with track(experiment=EXPERIMENT, out=out, seed=seed,
               clocks=[Step("updates", "optimizer updates")],
               config=dict(size=size, model=model, learning=learning,
                           observation=observation, decks=decks.digest)) as run:
        return self_play(
            run=run, seed=seed,
            updates=int(run.until("updates", default=updates)),
            model=model, learning=learning, observation=observation,
            decks=decks, game_errors=errors,
            streams=256, transitions=64, learning_seats="both",
            device="cuda", workers=4, threads=1,
            checkpoint_every=PROTOCOL.export_every,
            milestones=PROTOCOL.milestones,
        )


def runs(revision: str) -> list[Job]:
    # Existing launch declaration name and experiment-file organization stay.
    return [
        training(
            train, config=FrontierConfig(size=size, seed=PROTOCOL.training_seed),
            repo=ROOT, revision=revision, experiment=EXPERIMENT,
            max_hours=PROTOCOL.training_hours,
            max_usd=PROTOCOL.training_usd,
            max_attempts=1,  # Preserve Frontier's no-retry allocation.
            prepare=machine_setup(),
        )
        for size in ("small", "large")
    ]

# eval(state), play(match), report(): existing implementations retained.


# PROPOSAL — complete orchestration body; domain helper contracts below.
# Run is an Etude resource. It never receives a learner or optimizer.

def self_play(*, run: Run, seed: int, updates: int,
              model: ModelConfig, learning: DampedPolicyLearning,
              observation: ObservationConfig, decks: UniformDecks,
              game_errors: GameErrorPolicy, streams: int, transitions: int,
              learning_seats: str, device: str, workers: int, threads: int,
              checkpoint_every: int, milestones: tuple[int, ...]) -> Path:
    state_path = run.out / "state.pt"
    # Bind all semantic inputs once; mismatch refuses a resume.
    binding = training_binding(model, learning, observation, decks, seed)
    state = load_state(state_path, binding=binding, device=device)
    if state is None:
        state = new_state(model, learning, observation, seed=seed, device=device)

    # Resource lifetime only: native games, ledger and failure capture.
    # Seed/RNG restoration and cumulative counters stay in Manabot.
    with open_collection(
        decks=decks, observation=observation, state=state, seed=seed,
        streams=streams, workers=workers, threads=threads,
        learning_seats=learning_seats, game_errors=game_errors,
        evidence=run.out / "game-attempts",
    ) as games:
        while state.update < updates:
            if run.stopping():
                save_state(state_path, state, binding=binding)
                raise Stopped()  # Etude may retry within launch budgets.

            batch = collect(games, policy=state.policy, transitions=transitions)
            state, metrics = learn(state, batch, learning=learning)
            # Atomic save owns optimizer, EMA, RNGs and completed update count.
            save_state(state_path, state, binding=binding)
            run.log(metrics, updates=state.update)
            run.log_table("outcomes", games.take_outcomes(), updates=state.update)

            if (state.update % checkpoint_every == 0
                    or state.update in milestones):
                policy = export_checkpoint(run.out, state, binding=binding)
                run.checkpoint(policy, updates=state.update)

    # Immutable update-named export; repeat calls return the same bytes/path.
    policy = export_checkpoint(run.out, state, binding=binding)
    run.checkpoint(policy, updates=state.update, tags=("final",))
    return policy


# PROPOSED PUBLIC CONTRACT — learner-agnostic Etude tools.
# These are contracts, not an implementation of the scheduler.

def training(function, *, config: BaseModel, repo, revision, experiment,
             max_hours, max_usd, max_attempts, prepare) -> Job:
    """Validate the callable's Pydantic config and serialize it; declare a capped invocation."""


def track(*, experiment, out=None, seed, clocks, config) -> Run:
    """Open the current run; direct local use may supply out.

    A launch supplies identity, directory, deadline and destinations.
    Record source/host/attempt provenance without learner knowledge.
    Explicit experiment must agree with a launching experiment file.
    """


class Run:
    out: Path  # Working files restored before the callable starts.

    def until(self, clock, *, default):
        """Launch continuation target, or the callable's default."""

    def stopping(self) -> bool:
        """Stop request or deadline reserve based on observed polling intervals."""

    def log(self, values, **clocks) -> None:
        """Persist rows; first retry rows set the replacement boundary."""

    def log_table(self, name, rows, **clocks) -> None:
        """Persist structured measurements."""

    def checkpoint(self, path, *, tags=(), **clocks) -> None:
        """Record immutable bytes and deliver them; never decode the policy."""

# Launcher behavior, outside the recipe process:
# 1. Claim an attempt under a lease and the remaining run budgets.
# 2. Prepare the machine and restore the run's working files.
# 3. Invoke the recorded function with its arguments.
# 4. Carry files and tracking records during execution.
# 5. Planned-stop exit: settle attempt; retry only if policy/budgets allow.
# 6. Continuation: new run, copied parent out, new target, parent untouched.
# Eval scheduling consumes recorded checkpoints and experiment.eval(state).
# There is no stage interpreter, optimizer import or learner-state decoder.
