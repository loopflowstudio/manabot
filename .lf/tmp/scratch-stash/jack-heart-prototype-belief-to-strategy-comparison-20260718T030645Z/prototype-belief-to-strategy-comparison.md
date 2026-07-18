# Prototype: Belief-to-Strategy Comparison on the Shared Decision Surface

## Problem

The Game wave has a versioned `StudyArtifact` / `DecisionEvidence` contract
(GAM-4 KR done) and a canonical restored-decision fork path (RUL-4 done), but
no player-visible surface yet lets a human state a hidden-information belief,
obtain strategy from a pinned advisor, and compare the resulting action
probabilities, values, and uncertainty with explicit deltas. The unmet KR is:

> At the same canonical decision reached live or through Study, the same
> unified surface accepts at least two explicit viewer-safe belief scenarios,
> obtains strategy from one pinned advisor identity and compute class, and
> shows reproducible action-probability, value, or uncertainty deltas while
> facts, beliefs, and advice remain distinct.

This prototype builds that surface as the first AI-assisted play slice. It is
fixture-first against main: the canonical decision/retry parent (GAM-4's
publishable fork/Retry/return UI) is not yet landed, so the prototype works
against a checked, identity-pinned fixture produced from **real** flat-MC
search evidence and exposes a narrow adapter seam that can accept GAM-4's
contracts later. It does not duplicate or replace the StudyArtifact substrate.

Who benefits: a player who wants to understand *why* an action is right at one
decision by conditioning the advisor on different beliefs about what the
opponent holds. Why now: the contract and the canonical decision exist; the
missing piece is the unified player surface that proves the contract is
consumable end-to-end.

## The demo

Run `uv run pytest tests/etude/test_advice.py -q` and
`cd frontend && npm test -- --run advice` and
`cd frontend && npx playwright test advice.spec.ts`.

What appears: one reusable `DecisionAdvice` component renders at the same
`erd1` decision address in both the live play page (advisory, beside the
ActionPanel) and the replay/Study page (at the restored decision). The
component shows four visually distinct regions — **Facts** (the public board),
**Beliefs** (a selector for two named hidden-info scenarios with inferred
opponent starting ranges), **Advice** (per-action policy mass, expected match
points, and uncertainty from real flat-MC search), and **Deltas** (the explicit
probability, value, and uncertainty differences between the two scenarios).
Switching scenarios with pointer or keyboard updates the advice and deltas
live. The live instance submits only through the ActionPanel; the Study
instance shows the same advice at the same address with the same request
shape. Identity mismatch and missing-scenario requests return a typed
`unavailable` state, never fabricated advice.

## Approach

The prototype is four layers, each thin and substrate-respecting.

### 1. Real-evidence fixture (checked, versioned, identity-pinned)

A new fixture `protocol/fixtures/advice-curated-decision.json` produced by a
deterministic generator `scripts/generate_advice_fixture.py`. The fixture is a
`StudyArtifact` (the existing contract — no new protocol type) with **two
landmarks at the same `erd1` decision**, each carrying real flat-MC evidence
under a different pinned seed family. A top-level `scenarios` array maps each
landmark id to prototype presentation metadata (label, description, inferred
range, seed family, belief kind).

The pinned decision is the curated match's hero priority decision at ordinal 6
(UR Lessons vs GW Allies, seed 7, turn 7 precombat main: "Play Mountain" vs
"Pass priority"). The generator:

1. Plays the deterministic pinned match (reusing
   `scripts/generate_replay_fixtures._play_pinned_match`).
2. Forks the retained Study root at decision 6.
3. Runs `env.flat_mc_scores(worlds=1, rollouts=8, seed, max_steps=2000)` across
   two disjoint seed families (scenario A: seeds 101–116; scenario B: seeds
   201–216) — real engine search, not fabricated numbers.
4. Aggregates per scenario into `DecisionEvidence`:
   - `policy_mass`: fraction of the 16 seeds where each action is the argmax
     (a real action-probability distribution).
   - `search_value`: mean per-action win-probability across the 16 seeds
     (`expected_match_points`).
   - `visits`: total simulations across the seed family.
   - `sampled_world_robustness`: for each action, count of worlds (seeds) whose
     score exceeds 0.5, over 16 sampled worlds.
   - `uncertainty`: standard error of the per-action mean across the 16 seeds,
     `method = "flat-mc-seed-spread"`.
   - `provenance.producer`: `"flat-mc-search:v1:seeds-101-116"` (or `201-216`),
     `evidence_sha256`: sha256 of the canonical evidence payload.
5. Writes the fixture with the `StudyIdentity` pinned to the canonical replay
   id, match id, content/asset hashes, `model.id = "flat-mc-search-v1"`,
   `analysis_budget` recording the compute class (worlds=1, rollouts=8,
   sampled_worlds=16, rollouts_per_world captured in the budget id).

The fixture is **checked**: a test loads it, validates it as a `StudyArtifact`
through both the Pydantic model and the JSON Schema, and asserts the two
landmarks have distinct, non-uniform evidence with the recorded provenance.

### 2. Python adapter (`etude/advice.py`)

A narrow, pure-Python consumer of the fixture. It does NOT invoke search at
runtime — the fixture is the evidence. It validates identity and exposes
scenarios + deltas.

```python
class AdviceRequestIdentity(ProtocolModel):
    source_replay_id: str
    match_id: str
    advisor_id: str          # "flat-mc-search-v1"
    compute_id: str          # "1w-8r-16s"

class AdviceScenarioSummary(ProtocolModel):
    landmark_id: str
    label: str
    description: str
    inferred_range: str
    belief_kind: str

class AdviceResponse(ProtocolModel):
    status: Literal["ok", "unavailable"]
    reason: str | None              # "identity_mismatch" | "scenario_not_found" | "decision_not_found"
    address: str | None
    frame: ExperienceFrame | None
    offers: list[InteractionOffer]
    scenario: AdviceScenarioSummary | None
    evidence: DecisionEvidence | None
    deltas: dict[str, dict[str, float]] | None   # per-action {"policy_mass": ..., "search_value": ..., "uncertainty": ...}
    identity: AdviceRequestIdentity | None

def load_advice_fixture() -> AdviceArtifact: ...
def request_advice(address, scenario_id, identity) -> AdviceResponse: ...
def compute_deltas(left: DecisionEvidence, right: DecisionEvidence) -> dict: ...
```

`request_advice` fails closed: if the request identity does not match the
fixture's pinned identity, or the address does not match the fixture's
decision, or the scenario id is unknown, it returns `status="unavailable"`
with the specific reason and no evidence. The adapter cross-references the
`scenarios` metadata to the `StudyArtifact.landmarks` by landmark id and
validates that the artifact passes `StudyArtifact.model_validate` and
`assertViewerSafeStudyArtifact` (the existing TS-side twin is mirrored in
Python by calling the Pydantic validator + the opponent-hand check).

### 3. Server endpoint (`POST /api/advice`)

```python
@app.post("/api/advice")
async def request_advice_endpoint(payload: dict) -> dict:
    # parse AdviceRequestIdentity, address, scenario_id from payload
    # return request_advice(...).model_dump(mode="json")
```

One endpoint, one request shape, used by both live and Study. It loads the
fixture once at startup (the fixture is static and checked). No search runs at
request time — the prototype is fixture-first by directive.

### 4. Reusable component (`DecisionAdvice.svelte`)

A runes-mode Svelte 5 component (per the established interop rule) with props:

```typescript
interface DecisionAdviceProps {
  mode: 'live' | 'study';
  address: string;              // the erd1 decision address
  scenarios: AdviceScenarioSummary[];
  evidenceByScenario: Record<string, DecisionEvidence>;
  deltas: Record<string, Record<string, number>>;  // scenarioId -> per-action deltas
  offers: InteractionOffer[];   // the shared action vocabulary
  frame: ExperienceFrame;       // the facts
  selectedScenarioId: string;
  reducedMotion: boolean;
  onSelectScenario: (id: string) => void;
  onRequestAdvice: (address: string, scenarioId: string) => Promise<AdviceResponse>;
  onSubmitCommand?: (offerId: number) => void;  // live only; study omits
}
```

Four visually distinct regions, each with a stable `data-testid` and ARIA
role:

- **Facts** (`data-testid="advice-facts"`): the public board summary — turn,
  phase, life totals, hand sizes, battlefield counts. Rendered from the frame
  projection. No opponent hand identities (viewer-safe).
- **Beliefs** (`data-testid="advice-beliefs"`): a radio-group scenario
  selector with labels, descriptions, and inferred starting ranges. Keyboard
  navigable (arrow keys + Enter/Space). Each option carries
  `data-testid="advice-scenario-option"` and `data-scenario-id`.
- **Advice** (`data-testid="advice-advice"`): per-action rows showing policy
  mass (bar), expected match points, and uncertainty, each with
  `data-testid="advice-action-row"` and `data-action-id`. Bars respect
  `prefers-reduced-motion` (no transitions; instant width).
- **Deltas** (`data-testid="advice-deltas"`): the explicit per-action
  probability, value, and uncertainty differences between the selected
  scenario and the other, signed and color-coded.

In **live mode** (`mode="live"`), the component renders beside the ActionPanel.
The ActionPanel remains the only submission path (the live instance may submit
only its currently offered Command). The advice component's
`onSubmitCommand` is omitted or disabled — it is advisory-only. A footer notes
the pinned advisor identity and compute class.

In **study mode** (`mode="study"`), the same component renders at the restored
decision in the replay page. Same regions, same action vocabulary, same advice
request shape. The fork/Retry/return controls are NOT in this prototype — they
belong to GAM-4's substrate and the directive forbids duplicating them. A
narrow seam (`onRequestAdvice`) is the integration point for GAM-4's future
retry/compare contracts.

### Embedding

- **Play page** (`+page.svelte`): below the ActionPanel, a collapsible
  "Decision Advice" section renders `<DecisionAdvice mode="live" ... />` when
  a priority decision is active. It fetches the fixture's pinned decision via
  `/api/advice` on first open.
- **Replay page** (`replay/+page.svelte`): when the replay cursor rests on the
  pinned decision, the same `<DecisionAdvice mode="study" ... />` renders
  beside the board.

Both fetch from the same `POST /api/advice` with the same request body shape.
The component is the single shared surface; only `mode` and the surrounding
controls differ.

## De-risking

| Question | Finding | Impact on design |
|----------|---------|------------------|
| Can real flat-MC search run at the pinned Study decision and produce differing per-action scores across belief scenarios? | Yes. De-risked 2026-07-17: `env.flat_mc_scores(1, 8, seed, 2000)` at the forked decision-6 root produces seed-deterministic, reproducible scores that differ meaningfully across disjoint seed families. Scenario A (seeds 101–116): policy_mass [0.75, 0.25], mean [0.242, 0.211]. Scenario B (seeds 201–216): policy_mass [0.69, 0.31], mean [0.305, 0.336] (value flips toward Pass), favorable [0/16, 5/16]. | Use two disjoint seed families as the honest belief-scenario substrate. Aggregate 16 seeds per scenario into the existing `DecisionEvidence` shape. No fabricated numbers. |
| Is the same seed deterministic across forks and runs? | Yes. Same seed → byte-identical scores across two independent forks of the same retained root (verified). | Pin seeds in the fixture and provenance; the fixture is reproducible by re-running the generator. |
| Does the existing `StudyArtifact` contract permit two landmarks at the same `erd1` decision with different evidence? | Yes. `StudyArtifact.validate_bindings_and_privacy` iterates landmarks and validates each independently; it does not forbid duplicate `decision_id`. The `StudyLandmark.id` field distinguishes them. | Model belief scenarios as two landmarks with the same `decision_id` and distinct `id` + `evidence`. No new protocol type. |
| Can the advice fixture reuse the existing `study-v1.schema.json`? | Yes. The `artifact` field is a standard `StudyArtifact`; the `scenarios` metadata array is a prototype presentation layer validated by the adapter, not by the Rust-owned schema. | Do not extend the Rust JSON Schema. Validate the `artifact` with the existing schema and Pydantic model; validate `scenarios` with a small Pydantic model in `etude/advice.py`. |
| Does the structured offer surface at the forked branch match the frame's legacy offers? | No — `StudyBranch.structured_offers()` at decision 6 returns only `pass_priority`, while the frame carries two legacy offers (play_land, pass_priority). The flat-MC scores align with the legacy action space (2 entries), not the structured surface. | Use the frame's legacy `offers` as the shared action vocabulary (the existing fixture and live UI already do). Do not route advice through the structured offer surface in this prototype. |
| Is the live play page able to show advice at a decision that has not yet been recorded? | The live frame during play has no `erd1` address (the canonical replay is finalized only at game close). | The prototype's live surface loads the fixture's pinned decision (a completed-match decision) as the demonstration advice surface beside the live ActionPanel. The narrow seam (`onRequestAdvice`) is the future integration point for live-address advice once GAM-4 publishes the live decision address. |
| Will the component interop with the runes-mode stores without the frozen-DOM regression? | The established rule (2026-07-16) requires every new `.svelte` file to be runes mode. `DecisionAdvice.svelte` uses `$props`/`$state`/`$derived`/`$effect` only. | No legacy `$:` or `export let`. The e2e suite asserts DOM mutation on scenario switch. |
| Does `prefers-reduced-motion` need a server signal or is it client-only? | Client-only. The component reads `window.matchMedia('(prefers-reduced-motion: reduce)')` and disables bar transitions. The e2e test emulates the media query. | No protocol or server change for reduced motion. |
| Can the prototype avoid building the fork/Retry/return UI that GAM-4 owns? | Yes. The directive explicitly says the canonical decision/retry parent is not yet publishable and to not duplicate it. The Study side shows advice at the restored decision; fork/Retry/return is out of scope. | Study-mode `DecisionAdvice` is advisory + comparison only. `onRequestAdvice` is the seam; no fork controls. |

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Run flat-MC search live at request time | Real, dynamic advice; but the directive says fixture-first against main, do not fabricate dynamic claims or wait for the production belief player, and the live decision address is not yet publishable. | Fixed runtime cost, non-deterministic across engine builds, and violates the fixture-first directive. The fixture captures real evidence at a pinned identity; the seam accepts live search later. |
| Fabricate plausible evidence numbers | Fastest to build; but the directive requires "real policy/search evidence" and forbids "fabricated dynamic claims." | Violates the evidence-pinning requirement and the wave's evidence integrity (GAM-4 memory: checked fixtures exported for another replay must fail closed). |
| Extend the `StudyArtifact` protocol with a `belief_scenario` field | Clean protocol-level modeling; but the directive says "do not duplicate or replace that substrate" and the existing contract already permits two landmarks at one decision. | Reuse the existing contract; model scenarios as landmarks + a thin presentation-metadata layer. No Rust schema change. |
| Build a separate `/advice` route instead of embedding in play + replay | Cleaner demo isolation; but the directive requires the same component "in live play and Study." | Embed in both the play page and the replay page. The component is the deliverable; the routes are its hosts. |
| Use the structured offer surface as the action vocabulary | Forward-looking; but the structured surface at the pinned decision exposes only `pass_priority` while the frame and the live UI use the legacy two-offer surface, and flat-MC scores align with the legacy space. | Use the frame's legacy `offers` as the shared vocabulary to match the live UI and the existing fixture. |
| Add a full opponent-range visualization tool | Rich; but the directive forbids "generic range tooling." | Show a concise, honest, pinned inferred-range characterization per scenario as text. No interactive range editor. |

## Key decisions

- **Real evidence, not fabricated.** The fixture is generated by running
  `env.flat_mc_scores` at the forked Study root under two disjoint seed
  families. The scores are seed-deterministic and reproducible. The provenance
  records the exact seed family and compute class. This is the honest
  prototype the directive asks for.
- **Two landmarks, one decision, no new protocol.** Belief scenarios are two
  `StudyLandmark` rows at the same `erd1` address inside one `StudyArtifact`.
  The existing contract validates them. A thin `scenarios` metadata array
  carries the prototype's presentation layer (labels, descriptions, inferred
  ranges) without touching the Rust-owned schema.
- **Fixture-first, seam for later.** The `/api/advice` endpoint serves the
  static fixture. No search runs at request time. `onRequestAdvice` is the
  narrow seam that accepts GAM-4's live-address and retry/compare contracts
  when they publish.
- **Facts, beliefs, ranges, and advice are four distinct regions.** Each has
  its own `data-testid`, ARIA role, and visual treatment. The component never
  blends public board state with viewer beliefs or advisor output.
- **Live submits only its offered Command; Study compares only.** In live mode
  the ActionPanel is the sole submission path. In study mode the component is
  advisory + comparison; fork/Retry/return belongs to GAM-4 and is out of
  scope. Both use the same component, action vocabulary, and advice request
  shape.
- **Fail closed on identity mismatch.** `request_advice` returns a typed
  `unavailable` status with a specific reason and no evidence when the request
  identity, address, or scenario does not match the fixture. The component
  renders a typed-unavailable state, never fabricated advice.
- **Runes-mode component, reduced-motion and mobile first.** The component is
  runes mode (the frozen-DOM rule). Bar transitions respect
  `prefers-reduced-motion`. The layout collapses to a single column under the
  mobile breakpoint.

## Scope

- In scope: a checked, versioned advice fixture generated from real flat-MC
  evidence at a pinned world/viewer/advisor/compute/seed identity; a Python
  adapter that validates identity and exposes scenarios + deltas; one
  `POST /api/advice` endpoint; one reusable `DecisionAdvice.svelte` component
  embedded in the play page (live) and replay page (study) at the same
  decision address; two belief scenarios with inferred ranges and explicit
  per-action deltas; and focused pointer, keyboard, reduced-motion, mobile,
  viewer-safety, identity-mismatch, and typed-unavailable tests in Python,
  TypeScript unit, and Playwright e2e layers.
- Out of scope: live runtime search; the GAM-4 fork/Retry/return UI and its
  contracts; a new protocol type or Rust schema change; generic range
  tooling; chat; a second replay surface; client-side legality; broad Avatar
  content; spectator/pilot viewer roles; landmark ranking; and any change to
  the existing `study-curated-decision.json` fixture or the canonical replay
  schema.

## Done when

- `uv run python scripts/generate_advice_fixture.py` regenerates
  `protocol/fixtures/advice-curated-decision.json` deterministically, and the
  fixture validates as a `StudyArtifact` through both the Pydantic model and
  `protocol/study-v1.schema.json`.
- `uv run pytest tests/etude/test_advice.py -q` passes: fixture loads, two
  scenarios have distinct non-uniform real evidence, identity mismatch and
  missing-scenario and missing-decision requests return typed `unavailable`
  with no evidence, deltas are correct, and opponent hand identities are
  absent from every rendered frame.
- `cd frontend && npm run check` is clean and `npm test -- --run advice` is
  green: the `DecisionAdvice` component renders four distinct regions,
  scenario selection updates advice and deltas, and the TS twin of the
  adapter validates the fixture.
- `cd frontend && npx playwright test advice.spec.ts` is green across
  pointer, keyboard, reduced-motion, mobile viewport, viewer-safety (opponent
  hand not in DOM), identity-mismatch (typed-unavailable state rendered), and
  live-vs-study mode assertions.
- `uv run pytest tests/etude -q` and `cd frontend && npm run build` remain
  green (no regressions to the existing play/replay/study surfaces).

## Measure

This prototype does not tune the advisor, so it introduces no strength or
latency target. The outcome is binary and evidence-pinned:

- Two belief scenarios at the same `erd1` decision produce reproducible,
  distinct, non-uniform `DecisionEvidence` (policy mass, search value,
  uncertainty) from real flat-MC search at a pinned seed family.
- The deltas between scenarios are non-zero for at least one action across
  policy mass, search value, and uncertainty.
- The same component renders at the same address in live and Study with the
  same request shape, and identity mismatch fails closed with a typed
  unavailable state in every test layer.
