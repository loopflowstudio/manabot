# TLA pilot continuation — 2026-11-04 UTC

Jack Heart's accepted outcome remains preliminary GPU TLA self-play and paired
checkpoint evaluation on the frozen 134-deck trophy population, <=$1 per
allocation and <=$10 cumulative. Corpus/runners, Vast integration and an earlier
GPU learner milestone are delivered. An admitted trained checkpoint, complete
paired evaluation and Elemental-relative analysis remain outstanding.

## Current accepted design

Steer 3b0f8826 explicitly selects deterministic first-64 **legal** actions with
logging and cutoff, superseding the 512-action pilot. Steer f61beeb7 separately
requires native legality/payability and direct unavailable-X rejection without
state changes. Neither cutoff nor tensor capacity can legalize an invalid action.
V8 preserves input-v7 shared effects, card128, all 134 decks, recipe, seeds and
budgets. The fixed heuristic anchor retains its existing full native semantics;
learned candidates and the initial-policy anchor use the same cutoff as training.

`ObservationConfig.action_overflow=first-legal` is explicit in v8's regime,
protocol and checkpoints. Default `error` is omitted on serialization, preserving
frozen configurations. Native `policy_action_count` owns prefix selection;
Python and native buffered encoders consume it before bounded writes. Row i
remains native action/Command i. Native authority and Academy remain complete.
No pagination or global engine action filtering is introduced.

The existing `casting_mana_cost` repair owns selected-cost announcement/payment:
airbend {2} fixes printed X to zero; any-type-mana casts still pay printed X;
additional waterbend X survives a free non-X printed cost. New direct native
submission tests use current command bindings and full search-state witnesses.
Seven untapped basics pay precisely X=0..5 for {X}{U}{U}; X=6/255 reject unchanged.
Airbend and any-type-mana rejection checks also pass. No further rules repair
was exposed by these tests.

`input_buffer_config` still derives configurable axes from supplied arrays and
validates companion shapes. The cutoff does not alter any card, permanent,
focus, fact, relation or shared-effect capacity behavior.

Diagnostics count consumed observation decisions and overflow by decision kind,
with dropped totals and up to four full seed/deck/action-prefix samples carrying
original/retained/dropped counts. Console summaries occur at powers of two;
collector/game close writes exact aggregates, including zero overflow. Evaluation
retains the report in game receipts. Deliberate cutoff is distinct from a fatal
observation/game truncation. Scalar, native vector and sharded collectors use
the same path; no policy mask or browser filter duplicates legality.

Delete — do not maintain (completed): active `tla_pilot_v7.py` launcher and active
analysis/discovery imports, replaced by `tla_pilot_v8.py`; earlier completed cuts
removed `event_buffer_config`, buffer default overrides and printed-cost-only
X announcement. Frozen v1–v7 protocols, regimes and evidence remain unchanged.

## Frozen execution and spending

V5 Run `e8c76bc2c06f`, source `97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`,
ran on Vast RTX5090: three learner updates, 585 optimizer steps and 149260
transitions, then failed at 256 offered actions versus 64 before trained export.
The 256 offers were subsequently identified as illegal X0..255 during an airbend
cast of Wan Shi Tong. Source, failed attempt and initialization evaluation remain
frozen. The report retains 32 valid initial games (5 wins/27 losses), 256 missing
scheduled games and no improvement claim. All 114 v5 evidence files are published.

Prepared v6 Run `bcde93e9ff8c` at `ad2c3a010f53` and v7 Run `af3661790ac5`
at `b5406bb33abd98870af6ca07c3beed63a92b7c1c` were never submitted. Both are
superseded and must not launch. V7 artifacts and archive remain intact under
`.runs/mtg213/v7-continuation/` and `.runs/mtg213/runtime-b5406bb33/`.
The latter also passed the new retained-prefix rejection/parity diagnostic at 64
and 777 before the new cutoff direction arrived; it is evidence, not v8 authority.

Fresh 18:56:15 UTC inventory found no Vast instances, RunPod pods or unsettled
hardware. Accounted whole-rental cost remains $0.32710677365251912;
conservative historical reservations remain $3.9139916049382716/$10.
Earlier 18:44 quotes refused Vast's <=$0.81/hour ceiling and actual L4 capacity.
They establish no current availability and submitted nothing. Any v8 launch
requires fresh inventory, bills and an actual compatible quote.

Published isolated control client is `etude-sdk[s3]==0.2.14`; worker dependency
remains the recipe's 0.2.12. Credentials use nested Doppler `manabot/prd` then
`etude/prd`. V8 operational scripts under `.runs/mtg213/v8-continuation/` derive
from the retained v7 scripts; they preserve the prepared-run guard and budget
checks. V8 needs its own exact source freeze before quote/submission.

## Validation and remaining work

Check: directed debug `wan_shi_tong` 5 tests and native
`first_legal_policy_prefix` 1 test pass; rebuilt release extension; focused
Python cutoff/replay/collector/protocol checks pass. Gate owns broader affected
suites, full debug/Clippy, current-world admission and publication/delivery.

Remaining: freeze v8 source, reconcile/quote, execute one bounded GPU allocation,
retain actual learner/checkpoint evidence and teardown/billing, then complete the
paired evaluation and qualified Elemental-relative report. Historical missing
slots remain missing; no new result fills v5 denominators. No product or budget
approval is outstanding; compatible supply remains an external uncertainty.

Earlier continuation detail is retained at
`ebb42d0df85c6638d47a160f1f888e021622c102:scratch/tla-pilot-verify.md`;
uncommitted notes immediately before v8 consolidation are preserved under
`.runs/mtg213/pre-v8-notes/`. Wave memory is read-only in this implementation
step; its pre-existing unstaged changes are preserved.
