# MTG-500 · Finish manual mana payment

A manual payment must return the payer’s next legal choice, finish the cast, or reach the game’s normal outcome. One game waiting for a mana color currently holds every other game at the native step barrier.

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

## Where progress stops

- **A legal source disappears.** In `flow/play.rs`, `check_activation_terms` compares the player’s signed life directly with the ability’s life fee. At −1 this rejects the Room’s zero fee. A temporary switch to the existing `Game::can_pay_life` in `flow/effects.rs` restored exactly modes 1 and 3 on the checkout. The substitution left the Canyon failure intact.
- **An impossible source order is offered.** In `flow/mana.rs`, the payment model tracks printed life fees. Its output obligation omits Citadel’s damage instruction. It therefore proves a Citadel-first payment using a future Canyon activation. Live execution runs the damage through the source’s effect frame before exposing its color decision. By then Canyon’s fee is unaffordable. Canyon-first completes on the released engine.
- **The same decision is read indefinitely.** In `flow/tick.rs`, the trivial-choice loop increments its counter and continues even when the action list is empty. Nothing consumes the suspended decision. The next iteration rebuilds it from the same state. On tag `v0.2.6`, each mode check can enumerate battlefield triggers inside `after_mana_choice`. That enumeration is finite. HEAD shares a model across the modes and skips some trigger scans, but retains the empty-space loop.

## Implementation decision · made for review

Keep payment legality in the native engine. Choose an executable payment witness for each offered first action, including the life available **after** an activated mana ability resolves. Filtering the two impossible colors after Citadel has already tapped and dealt damage is too late. Capping trigger visits would discard independent Heartbeat bonuses and leave the suspended decision outstanding.

1. **Make the two source orders exact.** Start with the table above as a regression through normal cast, activation and mode submissions, with skipping off so the old engine fails an assertion immediately. Replace the duplicate fee comparison with `can_pay_life`. In `flow/mana.rs`, extend the existing ordered witness at the activation/output boundary for state-changing mana abilities. When the effect can affect a remaining payment, evaluate its native cost and effect transition on an isolated branch and continue affordability from the resulting resources. Share the transition with execution below offer validation so a probe never asks whether it may execute itself. A failed order backtracks to the next source or output. A damaging effect is legal even at lethal life. Prevention, replacements, life gain and genuine life fees use their existing rules and timing. Preserve reservations in nested callers. Branches with distinct effect outcomes must stay distinct in source grouping, memoization and direct affordability shortcuts. Ordinary fixed and modal taps retain the arithmetic path. Cover a prevented Citadel hit with a payable Canyon fee as the control against treating printed damage as a life cost.
2. **Close the no-progress path at the step boundary.** A singleton can be skipped only by executing its action successfully. An internally inconsistent, nonterminal suspended mana decision with zero offers should return the existing native `AgentError` before either skipping or publishing. Propagate it through positional and structured step completion. Strict vector stepping already propagates errors, and opted-in isolation already retains the failing slot’s frame (`agent/vector_env.rs`). The valid boards above must finish through normal rules in both skip settings, without invoking recovery. The invariant needs neither a clock nor a visited-board set.

The search question is the same for admission, displayed offers and execution-time revalidation. Preserve native action identities, source order, the chooser, individual bonus decisions, spend restrictions and receipts. The implementation replaces the raw life comparison, the effect-blind witness for the affected activations and the empty mana retry. Rayon, GPU collection, card definitions and serialized game formats need no changes.

## Acceptance demo

The implementation adds this focused card test alongside the existing Citadel damage and protection cases in `managym/cards/tests/cards/ody.rs`.

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards tarnished_citadel
```

The test exercises both branches of the Room’s blue/red choice and the legal orders for Canyon, checking taps, life, bonus mana, the spell and the subsequent state-based result. An offer read on a fork must leave the live game untouched. The failure assertion for the low-life Room runs before the skip-enabled case, so an old build reports the missing colors immediately.

For the collection check, restore the same pre-activation snapshot into one slot of a two-slot native vector with a healthy peer. Submit the displayed payment action with trivial skipping enabled. Both slots return, and the affected slot has the same choices or terminal outcome as scalar play. For the deliberately inconsistent suspended state, strict stepping returns an error and isolated stepping records just that slot’s failure. A small process deadline bounds the integration check. Paying the same position never depends on worker count, elapsed time or an action-search budget.
