# Manifesting Etude Fantasia

Plan date: 2026-07-16. Status: in progress — the `rename-etude` commit
(`f57cf63`) landed the repository rename, the README opening, the
`ETUDE_PLAY` namespace, and the paper retitle; per-item status is noted
below.

This plan makes the Etude Fantasia vision legible from the repository itself:
its name, its top level, its front door, and its documented paths. It follows
the July 15–16 direction shift — semantic kernel, certified play experience,
Study charter, measured research program — and the naming model adopted with
it.

## The naming model (canonical)

- **Etude Fantasia** — the full project name and the product experience.
- **Etude** — the short name wherever brevity or machine identity matters:
  repository, root directory, service namespaces, and ordinary prose after
  first mention.
- **manabot** — the agent, and the Python training library/CLI. Indefinite
  noun: you train *a* manabot.
- **managym** — the rules environment the agent lives in.

Boundary rule, for anything the four lines don't settle: **if it faces the
player, it is Etude; if it trains or evaluates the agent, it is manabot; if it
is the world, it is managym.** Consequences worth stating explicitly:

- The play server, experience protocol, curated packs, frontend, and their
  env/error namespaces take Etude names.
- Training infrastructure keeps manabot names — the wandb project, training
  presets, ops jobs, and the `manabot` CLI are correctly named today. The
  Intelligence wave's "Manabot Intelligence Initiative" is also still correct:
  Intelligence trains manabots.
- The engine keeps managym names.

Style and history policy:

- ASCII **Etude** everywhere, including prose. No accented Étude anywhere; a
  mixed corpus is churn waiting to happen.
- **Frozen evidence keeps its names forever**: experiment IDs (`exp-##`,
  `W2-###`), frozen contracts, receipts, and wandb run history. Project-level
  references update even in dated research documents — the rename commit
  moved `manabot-vs-phase.md` to `etude-vs-phase.md` because the comparison
  is about the project, which is now named Etude; the pinned commits inside
  it are the historical record. Living documents — README, charters, the
  paper, Linear initiative titles where the boundary rule says so — update.

## Design principles

1. **The filesystem tells the story.** The three names in the model should be
   the three top-level code directories: `etude/`, `manabot/`, `managym/`.
   A stranger running `ls` should learn the architecture.
2. **Every audience has a certified path.** `./scripts/play` set the standard:
   one command, structured failures, a clean-machine CI receipt. Playing,
   training, and (eventually) studying each deserve that tier.
3. **The research ledger is a named asset, not process residue.** The Phase
   comparison identified the explicit ledger — preregistered contracts,
   negative results, worlds discipline — as this project's unusual strength.
   The documentation should present it as such.
4. **Frozen evidence is immutable.** No rename or reorganization silently
   invalidates a pinned hash. Where reorganization intersects a frozen
   contract, the contract is versioned forward with an explicit migration
   note, never edited in place.

## Phase 0 — Canonize the name

Do this first so every later document references final names and URLs.

- **[landed]** Rename the GitHub repository → `loopflowstudio/etude`
  (redirects preserved; clone URLs updated in `f57cf63`).
- **[landed]** Rename the local root directory → `~/src/etude`.
- **[landed]** README opens with Etude Fantasia and introduces the naming
  model in prose.
- Add an explicit **Naming** section (the model above, verbatim, including
  the boundary rule) to the root README and to AGENTS.md. Agents author most
  commits here; an unwritten naming model will be violated.
- Audit Linear initiative titles against the boundary rule: Game and Study
  initiatives take Etude framing; Intelligence keeps Manabot framing; Rules is
  about managym's kernel. (Requires Linear access; not a repo change.)
- Leave the wandb project as `manabot` — training is the agent's domain.

## Phase 1 — Make the top level tell the story

### `gui/` → `etude/`

`gui/` is the one directory whose name is now actively wrong: it is the
authoritative experience server for a finished game — experience protocol,
presentation, study protocol, curated packs — not a debug GUI. The Game wave
already renamed `wave/gui` → `wave/game`; the code should follow, and the
package should carry the product's name.

Grounded touch list:

- `gui/*.py` internal imports; `scripts/play.py`; `tests/gui/` →
  `tests/etude/`; `manabot/sim/teacher1_evidence.py` (imports
  `gui.experience_protocol`, `gui.server`, `gui.trace`).
- `pyproject.toml`: `packages = ["manabot"]` becomes
  `packages = ["manabot", "etude"]` (gui is not packaged today; the product
  server should be), and `known-first-party` gains `etude`.
- **Frozen-contract hazard (decide before landing):** the Teacher-1 pilot
  contract (`experiments/contracts/w2-234-teacher1-pilot-v1.json`) pins
  `pilot_source_sha256` and `experience_protocol_sha256`, and
  `teacher1_evidence.py` imports from `gui`. Either (a) run and close the
  Teacher-1 pilot before this rename, or (b) cut
  `w2-234-teacher1-pilot-v2.json` re-pinning hashes with an explicit
  migration note. Never edit v1 in place.

### Env and error namespaces

Split by the boundary rule:

- **[landed]** Play-facing `MANABOT_PLAY_*` env vars and the
  `MANABOT_PLAY_ERROR` structured-error prefix are now `ETUDE_PLAY_*` /
  `ETUDE_PLAY_ERROR` (`f57cf63`).
- Training-facing `MANABOT_*` vars (`manabot/infra/hypers.py`,
  `manabot/verify/store.py`, `ops/`) stay — they are the agent's.

### Root cleanup

- `Dockerfile` and `entry.sh` move under `ops/` (reconcile with the existing
  `ops/Dockerfile`).
- `scratch/` (empty) and `dist/` leave the tracked root or get gitignored.
- `frontend/` stays put this phase — tooling paths are sticky — but is
  documented everywhere as Etude's client. Optional later move to
  `etude-web/` once the dust settles.

Resulting top level: `etude/ manabot/ managym/ frontend/` (the system),
`protocol/ content/ conformance/` (the contracts between them),
`experiments/ wave/ docs/ paper/ WORLDS.md` (the research ledger),
`ops/ scripts/ tests/` (operations).

## Phase 2 — The front door

Rewrite the root README top to bottom. Today it opens "a reinforcement
learning framework… using PPO" and reaches AWS/wandb credentials before
revealing that a playable game exists. New shape:

1. **`# Etude Fantasia`** — one paragraph: a fantasy card game you play
   against a learning agent, and a study experience that turns finished games
   into understanding. Introduce "Etude" as the short name in the first
   sentence. One screenshot from the versioned release visual references.
2. **Play** — `./scripts/play`, the three prerequisites (uv, Node, Rust), and
   what you get: the curated matchup against a trained manabot, offline-capable
   and recoverable.
3. **Study** — one honest paragraph on what exists and what the Study charter
   is building.
4. **Train a manabot** — three lines and a link to `manabot/README.md`.
5. **The research ledger** — `experiments/`, `WORLDS.md`, `wave/`, `paper/`,
   framed as an open lab notebook with preregistered contracts and recorded
   negative results.
6. **Map of the repository** — one table, one line per top-level entry.
7. **Naming** — the four-line model.

Everything else the README currently holds moves to where its reader looks
for it: engine architecture to `managym/README.md`, training detail to
`manabot/README.md`, style guides to AGENTS.md/CONTRIBUTING.

## Phase 3 — A document for every reader

- **`manabot/README.md`** (trainer): quickstart with the new local preset
  (Phase 4), preset table, a one-paragraph "you are in world w2 and these are
  the live baselines" orientation, wandb setup for real runs, sim usage,
  links to `WORLDS.md` and the ledger.
- **`managym/README.md`** (engine developer): the architecture section moved
  from the root README, the semantic-kernel overview
  (`docs/research/semantic-kernel.md`) as the front pointer, rebuild
  instructions.
- **`etude/README.md`** (product developer): the authority model (server
  authoritative, client renders), experience protocol v1 summary with links
  to `protocol/` schemas, curated packs, study protocol, and the
  clean-machine proof.
- **`docs/README.md`**: an index — architecture / research / rules /
  benchmarks — one line each.
- **`protocol/README.md`** expansion: versioning policy, the three-language
  certification story, how a v2 gets proposed.
- **`experiments/README.md`**: the ledger index — a table of every exp-## and
  W2-### with its one-line result, negative results included ("bag encoder
  exactly 50%; relational encoder 82.1%"). This is the single page that makes
  the research program legible to an outside researcher.
- **`wave/README.md`**: explain the wave/GOAL/MEMORY process itself — the
  process is part of what's worth showing.

## Phase 4 — Certified journeys (the centerpiece)

Extend the `./scripts/play` standard — one command, structured failure,
CI-receipted clean-machine proof — to the full loop.

- **Journey 1: Play.** Done. Keep the receipt green.
- **Journey 2: Train a manabot.** `uv run manabot train --preset local`: runs
  on a laptop (CPU/MPS), requires no credentials (wandb offline or disabled),
  bounded to roughly ten minutes, and emits a usable checkpoint. Certified by
  a `scripts/verify-training` analog of `verify-clean-machine` with its own
  CI receipt. Today training is gated behind Ubuntu/AWS + wandb; this is the
  gap between "the agent is the product" and "the agent is something only the
  authors can touch."
- **Journey 3: Face your own manabot.** `./scripts/play` accepts a checkpoint
  for the villain seat. This closes the loop that *is* the vision: play →
  study → train → play against what you trained. Etude Fantasia becomes a
  game whose opponent you can grow. No other change in this plan manifests
  the vision as directly.
- **Journey 4: Study (deferred).** When the Study wave ships decision
  landmarks, the journey gets the same treatment: one command from a finished
  game to a guided review, with a receipt.

Each certified journey earns a first-screen entry in the README.

## Phase 5 — The open lab notebook

Ambitious tier; sequence after Phases 0–4.

- Publish the ledger — `experiments/`, `WORLDS.md`, wave charters, selected
  `docs/research/` — as a static site: *Etude: an open laboratory for game
  intelligence*. The material already meets a publishable bar; it is only
  undiscoverable.
- **[landed]** Reconcile `paper/` with the naming model (`f57cf63` retitled
  the draft and built artifact to `etude.pdf`).

## Phase 6 — Naming conformance as a gate

Fitting the repo's fail-closed culture: `scripts/check_naming.py`, run in CI,
that rejects new `from gui`/`import gui`, any accented "Étude", play-facing
`MANABOT_` identifiers outside a frozen-history allowlist, and un-uv'ed
Python invocations in docs. Cheap to write, and it makes Phase 0's canon
self-enforcing.

## Sequencing and risks

| Order | Work | Risk to manage |
|---|---|---|
| 1 | Phase 0 naming canon + repo rename | Do first so all new docs use final URLs |
| 2 | Phase 2 README + Phase 3 docs | None; pure docs. Can reference `etude/` as "landing in the next PR" if it races Phase 1 |
| 3 | Phase 1 `gui/`→`etude/` + env prefixes | **Teacher-1 frozen contract**: close the pilot first or version the contract to v2; re-run clean-machine proof in the same PR |
| 4 | Phase 4 training journey + villain checkpoint | New code; needs Intelligence-wave coordination on preset/checkpoint format |
| 5 | Phase 6 conformance gate | After renames land, so the allowlist is small |
| 6 | Phase 5 lab notebook | Independent; any time after Phase 3 |

Never renamed: experiment IDs, frozen contracts and receipts, wandb history,
`manabot`/`managym` themselves.

## Success criteria

- A stranger on a clean machine is **playing Etude Fantasia in ≤ 5 minutes**
  (already proven), **training a manabot in ≤ 15 minutes**, and **facing
  their own manabot the same session** — each step one documented command,
  each certified by a CI receipt.
- The README's first screen contains the name, an image, and the play
  command — no RL vocabulary before the game.
- `ls` at the root teaches the naming model without commentary.
- Every top-level directory has a README or a one-line entry in the root map.
- `grep -r "from gui"` returns only history; the naming gate is green in CI.
- An outside researcher can find and understand the experiment ledger from
  the README in two clicks.
