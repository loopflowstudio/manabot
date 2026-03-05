# 01: Coverage Baseline + Rule-Cited Test Harness

## Problem

The engine implements ~12 rule families (priority, lands, combat, SBA, mana, cleanup) across ~4,100 lines of Rust, but none of this is attributed to Comprehensive Rules references. There are 14 tests total (11 integration + 3 mana unit), all behavioral smoke tests — none test a specific CR rule or negative path. We can't answer "what rules are implemented and tested" without reading every line of `game.rs`.

This blocks the entire rules wave: every subsequent stage needs a test pattern, a coverage map, and citation conventions before it can ship attributed work.

**Wave goals advanced:**
- "Every implemented rule family has focused CR-cited trace tests (plus negative paths)" — establishes the pattern
- "Rule expansion stages remain independently shippable" — creates the scaffold future stages build on

## Approach

Three deliverables, one PR:

### 1. Coverage artifact: `docs/rules_coverage.yaml`

YAML file mapping CR rule families to implementation status. Each entry:

```yaml
- rule: "CR 305 — Lands"
  status: implemented_tested  # not_started | scaffolded | implemented | implemented_tested
  code_refs:
    - managym/src/flow/game.rs:586-602   # play_land, can_play_land
    - managym/src/flow/game.rs:172-174   # lands_played counter
    - managym/src/flow/game.rs:490-496   # land action generation
  test_refs:
    - managym/tests/rules/cr_305_lands.rs
  notes: "One land per turn, main phase only, stack empty. No extra-land effects."
```

Status ladder:
- `not_started` — rule family not in the engine
- `scaffolded` — stubs/types exist, no behavior
- `implemented` — behavior works, no rule-specific tests
- `implemented_tested` — CR-cited tests cover happy + negative paths

(Defer `parity_validated` and `documented_deviation` — those matter when we have enough coverage to compare against a reference implementation. Not yet.)

Initial families to map (~12 entries):
| Family | CR | Status target |
|--------|-----|--------------|
| Game start / opening hand | 103 | implemented_tested |
| Mana production & payment | 106 | implemented_tested |
| Priority & passing | 117 | implemented_tested |
| Lands | 305 | implemented_tested |
| Casting spells (sorcery speed only) | 601 (subset) | implemented_tested |
| Combat: declare attackers | 508 | implemented_tested |
| Combat: declare blockers | 509 | implemented_tested |
| Combat damage | 510 | implemented_tested |
| End of combat | 511 | implemented |
| Cleanup step | 514 | implemented_tested |
| SBA: lethal damage | 704.5g | implemented_tested |
| SBA: life <= 0 | 704.5a | implemented_tested |
| SBA: empty library draw | 704.5b | implemented_tested |
| Turn structure | 500-514 | implemented |

### 2. CR citations in existing code

Add `// CR xxx.x` comments to existing implementation code at the point where a rule is enacted. Not decorative — each citation marks a specific rule being applied.

Target files:
- `managym/src/flow/game.rs` — bulk of citations (priority, lands, combat, SBA, cleanup, casting)
- `managym/src/flow/turn.rs` — turn/phase/step structure
- `managym/src/state/mana.rs` — mana payment rules

Convention:
```rust
// CR 305.1 — Play land from hand during main phase, stack empty
fn can_play_land(&self, player_id: PlayerId) -> bool {
```

### 3. Rule test structure + backfill

Create `managym/tests/rules/` with scenario helpers and one file per CR chapter.

#### Test harness: `managym/tests/rules/helpers.rs`

A small scenario builder on top of the existing `Game` API. Not a full trace runner — just enough to make rule tests concise and uniform.

```rust
pub struct Scenario {
    game: Game,
}

impl Scenario {
    pub fn new() -> ScenarioBuilder { ... }
}

pub struct ScenarioBuilder {
    decks: Vec<BTreeMap<String, usize>>,
    seed: u64,
    skip_trivial: bool,
}

impl ScenarioBuilder {
    pub fn deck(mut self, cards: BTreeMap<String, usize>) -> Self;
    pub fn seed(mut self, seed: u64) -> Self;
    pub fn skip_trivial(mut self, skip: bool) -> Self;
    pub fn build(self) -> Scenario;
}

impl Scenario {
    // Advance game state
    pub fn advance_to_step(&mut self, step: StepKind);
    pub fn advance_to_phase(&mut self, phase: PhaseKind);
    pub fn take_action_by_type(&mut self, action_type: ActionType) -> bool;
    pub fn pass_priority(&mut self);
    pub fn step_action(&mut self, index: usize);

    // Assertions
    pub fn assert_action_available(&self, action_type: ActionType);
    pub fn assert_action_not_available(&self, action_type: ActionType);
    pub fn assert_life(&self, player: usize, life: i32);
    pub fn assert_zone_size(&self, player: usize, zone: ZoneType, expected: usize);
    pub fn assert_game_over(&self);
    pub fn assert_winner(&self, player: usize);

    // Accessors
    pub fn game(&self) -> &Game;
    pub fn game_mut(&mut self) -> &mut Game;
    pub fn life(&self, player: usize) -> i32;
    pub fn zone_size(&self, player: usize, zone: ZoneType) -> usize;
    pub fn action_space(&self) -> &ActionSpace;
    pub fn current_step(&self) -> StepKind;
}
```

This builder wraps `Game` directly — no JSON, no deserialization, no new dependencies. Tests get type checking, IDE navigation, and debugger support. If we want JSON traces later (stage 02+), the builder can grow a `from_json()` constructor without changing existing tests.

#### Test files

```
managym/tests/rules/
    mod.rs          -- pub mod declarations
    helpers.rs      -- ScenarioBuilder + Scenario
    cr_103_game_start.rs
    cr_106_mana.rs
    cr_117_priority.rs
    cr_305_lands.rs
    cr_508_attackers.rs
    cr_509_blockers.rs
    cr_510_combat_damage.rs
    cr_514_cleanup.rs
    cr_601_casting.rs
    cr_704_sba.rs
```

Each file contains 2-5 focused tests. Example:

```rust
// cr_305_lands.rs
// Tests for CR 305 — Lands

use super::helpers::*;

/// CR 305.1 — A player may play a land during a main phase of their turn
/// when the stack is empty.
#[test]
fn cr_305_1_play_land_main_phase() {
    let mut s = Scenario::new()
        .deck(mountain_deck())
        .deck(mountain_deck())
        .seed(42)
        .skip_trivial(false)
        .build();

    s.advance_to_step(StepKind::Main);
    s.assert_action_available(ActionType::PriorityPlayLand);
    let hand_before = s.zone_size(0, ZoneType::Hand);
    s.take_action_by_type(ActionType::PriorityPlayLand);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before - 1);
}

/// CR 305.2 — A player can normally play one land per turn.
#[test]
fn cr_305_2_one_land_per_turn() {
    let mut s = Scenario::new()
        .deck(mountain_deck())
        .deck(mountain_deck())
        .seed(42)
        .skip_trivial(false)
        .build();

    s.advance_to_step(StepKind::Main);
    s.take_action_by_type(ActionType::PriorityPlayLand);
    // After playing one land, no more land actions available
    s.assert_action_not_available(ActionType::PriorityPlayLand);
}

/// Negative: Cannot play land during combat
#[test]
fn cr_305_negative_no_land_during_combat() {
    let mut s = Scenario::new()
        .deck(mountain_deck())
        .deck(mountain_deck())
        .seed(42)
        .skip_trivial(false)
        .build();

    s.advance_to_step(StepKind::BeginningOfCombat);
    s.assert_action_not_available(ActionType::PriorityPlayLand);
}
```

Target: ~30-40 tests across 10 files, covering every implemented rule family with at least one happy-path and one negative-path test each.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| JSON trace files + runner | Declarative, scales to hundreds of scenarios, non-programmers can author | Adds deserialization layer, new dependency (serde for test DSL), harder to debug, overkill for 30 tests. Can add JSON frontend to the builder later. |
| Inline `#[cfg(test)]` modules per source file | Collocated with implementation | Rule tests cross module boundaries (a land test touches game.rs, zone.rs, mana.rs). Integration-style tests in `tests/` fit better. Unit tests stay inline where they exist (mana.rs). |
| Snapshot testing (insta crate) | Auto-generates expected output | Wrong tool — we need to assert specific rule properties, not match full output snapshots. |

## Key decisions

**Execution package (validated): One more big push.**
Land the full stage in one coherent PR:
- Coverage artifact (`docs/rules_coverage.yaml`)
- CR citations in existing implementation code
- `tests/rules/` helper harness + broad rule backfill (~30-40 tests)

Defer JSON trace ingestion, parity/deviation statuses, and training metrics to later wave stages.

**Rust-first, not JSON-first.** The wave vision mentions a trace-based harness. We honor that direction by building a scenario builder that *could* be driven by JSON, but start with typed Rust. 30 tests don't need a data format. When we hit 100+, we can add `Scenario::from_json()` without rewriting existing tests. Starting with JSON now would mean building and debugging a mini-DSL before writing a single rule test.

**Integration tests in `tests/rules/`, not unit tests in `src/`.** Rule tests inherently exercise multiple modules (game flow + zones + mana + permanents). Rust's `tests/` directory gives us access to the public API without `pub(crate)` gymnastics.

**Flat status ladder (4 levels, not 6).** Dropped `parity_validated` and `documented_deviation` for now. Those statuses make sense when comparing against a reference engine or when we intentionally diverge. Neither applies yet. Can add them when they're needed — YAML is additive.

**One PR, not three.** Coverage artifact + citations + tests are tightly coupled. Splitting them creates partial states where citations exist without tests to verify them. Ship together.

**No training baseline metrics in this stage.** The original spec included branching factor / action space / episode length metrics. Moved to a separate concern — this stage is about rule coverage visibility, not training diagnostics. Training smoke tests start in stage 02.

## Maintenance contract (to prevent drift)

1. **Helper scope guard:** `tests/rules/helpers.rs` is a thin wrapper over public `Game` APIs; no duplicated game logic.
2. **Citation anchor:** CR comments/tests cite the snapshot in `docs/rules/MagicCompRules-20260227.pdf`.
3. **Coverage sync rule:** Any PR adding/changing rule tests in this wave updates `docs/rules_coverage.yaml` in the same diff.

## Scope

**In scope:**
- `docs/rules_coverage.yaml` with all currently implemented families
- CR citations added to `game.rs`, `turn.rs`, `mana.rs`
- `managym/tests/rules/` directory with helpers and ~10 test files
- ~30-40 rule tests covering existing behavior (happy + negative paths)
- Citation naming convention: test names prefixed `cr_NNN_`, comments cite full CR reference

**Out of scope:**
- JSON trace file format (future, builder can support it later)
- Training baseline metrics (separate concern)
- New rule implementations (this stage tests what exists, doesn't add mechanics)
- Multiplayer (wave scope: two-player only)
- `parity_validated` / `documented_deviation` statuses

## Done when

```bash
# All tests pass including new rule tests
cargo test -p managym

# Coverage artifact exists and is parseable
cat docs/rules_coverage.yaml | head -20

# Rule test files exist with CR citations
ls managym/tests/rules/cr_*.rs

# CR citations exist in implementation code
grep -r "// CR " managym/src/ | wc -l  # expect 15+
```

Observable: `docs/rules_coverage.yaml` answers "what's implemented and tested" at a glance. Every implemented rule family has at least 2 tests (happy + negative). Implementation code has CR citations at the point of rule application.
