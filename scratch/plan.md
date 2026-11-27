# MTG-500 · Finish manual mana payment

A manual payment must return the payer’s next legal choice, finish the cast, or reach the game’s normal outcome. The reported stall held every game at the native step barrier.

Pass 1 (`1c521e300`) propagated empty, forced and automatic decision errors and applied the zero-life cost rule. The Room at −1 offers blue and red, isolated stepping poisons only the malformed slot under both skip settings, and all 23 focused debug tests passed.

Pass 2 (`a9359c03b`) carries signed life through mana, checks trailing fees and records source coupling. All 78 focused debug checks passed, including the four Heartbeat positions, the announced and Ward fees, and the read-only fork. The 512-game debug step benchmark was stopped at 4m41s before it emitted a digest. The release Engine Budget comparison remains in pass 6.

The first pass 2 attempt found the counterexample below (`34092c814`, test `tarnished_citadel_and_the_great_henge_change_another_mana_source` in `managym/cards/tests/cards/ody.rs`) and changed no production code. The concept review (`24f1ec2f3`, `scratch/concept-review.html`) replanned around it.

Pass 3 (`abfabece991`) owns the announced mana activation and its priced fee, reads dynamic output at completion, reserves paid taps from the owed cost, classifies Treasure against battlefield reads, and bumps the match-state hash to v35. Twelve focused debug checks passed, including live and snapshot-restored Ogre, Henge repricing, and conditional Treasure.

Pass 4's finite-resource contract (`ab008535c`) shares one mana-ability predicate, routes board-growing and counter-swapping abilities through the stack, and reserves the pending activation's counters and source presence. The 48 focused debug checks passed. Catalog parity held across printed, token and granted definitions, with suffixes on Tarnished Citadel and The Great Henge only.

Pass 4a (`494fb4c5e`) retains counter and sacrifice costs before computing mana output and walks the saved frame for obligations and reservations. All 56 focused debug checks passed across the mana and retained-batch fixtures, including live and restored replacement orders, counter and sacrifice outputs, and nested cast and resolution payments.

Pass 4b (coupled regime) stopped at five counterexamples and moves to a separate Task (eighth review 1a, below). Each is a committed debug test that records the defect at the point of failure: `0d7bb6743` and `a90fd7109`, `de9787c04` (closed by design 5), `da29dbbdc` (closed by pass 4a) and `710163f05` (fixed tap opens a bonus choice). Their full analysis and designs 1, 4 and 7 are in `7cb3655f1:scratch/plan.md`.

Pass 5 (`12950aefb`) removes the provisional coupling flags, defers automatic commitment through the last mana bonus, and puts colored convoke in the shared witness. All 65 focused debug checks passed, including Canyon-first beside Angel, choosing which mana to spend after Heartbeat, and the Room into Guardian cast in manual and automatic modes.

## Design · made for review

Headless choices following each concept review's recommendation; reversible internal engine work, approved by nobody.

1. **Exact life.** Signed `LifeBudget { life: i32, cant_change: bool }`; activation `life_cost` checked against the running budget; post-mana life (announced `payment.life()`, cast `life_alternative`, resolution tail life) is a trailing `Obligation::PayLife(u32)` at completion (CR 601.2h). Built in pass 2.
2. **Closed monotone mana-ability contract** (sixth review). One `mana_parts()` predicate; finite resource; suffixes limited to `GainLife`, `DrainLife`, untargeted `LoseLife`/`DealDamageToPlayer`; pending `remove_counters` reserved on `ManaUpfront`; catalog CR 605.1a parity in CI. Built in `ab008535c`.
3. **Phase-accurate mana activation continuation** (seventh review 1a, 3a, 4a). `FrameCompletion::ManaOutput(Box<PendingManaOutput>)` evaluates output after counter and sacrifice costs with captured `chosen` LKI; reservation and obligation walks follow `caller` and `resume`; hash stays v35. Built in `494fb4c5e`.
4. **Fixed suffix life delta** (eighth review 1a). Probe fixed life gain and self-damage through replacements in an isolated fork, even with life-reading statics on the board. The coupled regime moves to the Task below. Built in pass 5.
5. **Finish the production before committing mana** (eighth review 3a). Keep the original output pooled while a bonus decision is pending. At completion, spend only uncontested units automatically and retain any remaining spending choice. Built in pass 5.
6. **Colored convoke in the witness** (second and eighth review 4a). A source can tap for an owed pip of its color on the same search path as mana and generic help. The transition reserves that tap from later obligations. Built in pass 5.

Preserve native action identities, source order, the chooser, individual Heartbeat bonus decisions, spend restrictions and receipts. Keep replay and action formats. The `PaymentScan` debug rebuild check must keep holding.

## Remaining passes

6. **Collection check and acceptance.** Restore a pre-activation snapshot into one slot of a two-slot native vector with skipping on; both slots return and match scalar play. Acceptance demo, debug nextest for touched crates, release engine budget comparison, and the `managym/README.md` mana-payment section for the built design.

## Reproduced on a TLA Limited board

Cast **Sokka, Bold Boomeranger** for {U}{R} with an empty pool, two untapped lands and **Heartbeat of Spring**. Activate **Tarnished Citadel** for colored mana first. Its ability deals 3 damage to its controller. Each land that produces mana triggers Heartbeat once. Citadel, Canyon and Heartbeat are in the TLE bonus sheet used in TLA Limited (`managym/cards/tests/fixtures/tle-bonus-sheet.jsonl`).

| Second land | Starting life | What the unpatched engine offers after Citadel | Correct result |
|---|---:|---|---|
| Airship Engine Room | 2 | Empty color decision at −1 life | Blue and red both complete the cast. The Room supplies the other color. State-based actions check for a loss after payment. |
| Airship Engine Room | 3 | Blue and red at 0 life | Same two choices. |
| Sunbaked Canyon | 3 | Empty color decision at 0 life | Only Canyon can go first. Pay 1 life, choose red, then use Citadel for blue. The cast completes and the payer loses normally. |
| Sunbaked Canyon | 5 | Blue | Either land can go first. The payer finishes at 1 life, with {U}{R} floating from Heartbeat and Sokka on the stack. |

The public 0.2.6 wheel and native checkout `b611abb3d7` both returned the two empty decisions with `skip_trivial=false`. On the wheel, enabling skipping made both failing steps exceed a two-second subprocess deadline. The adjacent controls returned. The original Linux report provides a sampled stack and no replay, seed or saved board. These positions reproduce the reported call path and stall independently. [Issue 628](https://github.com/loopflowstudio/manabot/issues/628)

Run the bounded release probe from the repository root. It prints the four mode lists in table order, `[]`, `[1, 3]`, `[]`, `[1]`.

```sh
uv run --no-project --python 3.13 --with managym==0.2.6 python scratch/repro_mana_payment.py
```

## Acceptance demo

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards tarnished_citadel
```

The test exercises both branches of the Room’s blue/red choice and the legal orders for Canyon, checking taps, life, bonus mana, the spell and the subsequent state-based result. The low-life Room failure assertion runs before the skip-enabled case, so an old build reports the missing colors immediately. Paying a position never depends on worker count, elapsed time or a search budget.

## Concept review disposition

Reviews two through seven are designs 1–3 and the deferred findings (detail in `7cb3655f1:scratch/plan.md`). Eighth review (`65036139e`, `scratch/concept-review.html`): 1a is design 4 and the split below; 2a puts the fixed-tap continuation first in the split Task; 3a is design 5; 4a is design 6. All made for review. Its pass 4a verification found no defect; its handoff invariant is recorded in the split Task.

## Deferred findings

- **Coupled mana payment** (separate Task; eighth review 1a/2a, made for review). Payment steps that change another source's output, cost or triggers. Slices, in order:
  1. Fixed-tap bonus continuation (`710163f05`): evaluate `tap_bonuses` with the permanent tapped (CR 106.12a); classify fixed sources as `FixedOutput::{Exact(Mana), Choice(ManaChoices)}` so a multi-type fixed source beside Heartbeat is not dropped by `fixed_source_mana` (`outputs.len() > 1` returns `None`); `tap_for_mana_action` takes the caller's frame and resumes through the existing `ManaProduction::Bonus`; only `Exact` sources enter automatic batches (`produce_mana`, `pay_mana_upfront`, `preview_tap`). No new decision variant or hash change.
  2. Closed positive whitelist `BoardArithmeticProfile` with roles Invariant/Sensitive/Mutator (fifth review design 4).
  3. Coupled fork search with pure-model handoff, memoized by canonical payment state (second review design 1). It must step `OrderReplacement` natively until `ManaChoice`: `payment_obligations_owing` omits an in-flight `ManaOutput`'s own output while suspended there.
  4. Payer `OrderReplacement` filtered to completable orders during an enclosing payment (seventh review design 7).
  Acceptance: `0d7bb6743`, `a90fd7109`, review case 2 (Season/Caradora), the fifth-review boards and the frozen-list boards (Angel of Vitality, Raucous Audience; `7cb3655f1:scratch/plan.md`) become live completions; ordinary TLA boards stay fork-free; `{15}` with 10 Forests rejects quickly in debug.
- **Own announced non-mana activations** (third review choice 3a, made for review; 3b would fold it in here). `Operation::Activation` / `ActivatedTargeting` re-read `activated_abilities.get(ability_index)` in `prepare_activated_ability` (`flow/play.rs` ≈1871/1986) after mana is spent (`flow/payment.rs` ≈1413). A grant that ends while paying gives `AgentError("invalid ability index")` or shifts to another ability. Needs owned capture in `flow/play.rs` begin/pay, `flow/payment.rs`, and the hash; a separate Task.

## Docs

- `docs/engine.md` (`### Forced decisions`): `Game::tick` returns `Result<bool, AgentError>`; empty non-terminal spaces and failed forced/automatic actions error under both skip settings; isolated vector slots poison alone. Built in pass 1.
- `managym/README.md` describes the signed life budget and the order of fees. Pass 6 documents the completed payment model and grouped help.
