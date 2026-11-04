# MTG-213 preliminary TLA trophy pilot

Jack Heart authorized this pilot on 2026-11-03 and subsequently authorized a
**$10 total paid-compute cap**, including setup, failures, retries, training
and evaluation. Enabling changes may ship after required checks. This protocol
was written before any pilot scoring. It is an engine/learning diagnostic.

## Population and provenance

Deck data comes from [17Lands public datasets](https://www.17lands.com/public_datasets),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 17Lands does
not endorse this pilot. The trophy API currently forbids off-site reuse; the
public TLA PremierDraft game dump is the permitted source. No trophy-API data
enters the corpus.

`experiments/data/mtg213/corpus-v1/` retains the exact 47,153,098-byte gzip,
retrieval metadata/digest, and `corpus.json`. The public dump was last modified
2026-01-10. Select the first 1,000 events with seven distinct recorded wins in
file order (67,109 rows scanned), using each seventh-win game's main deck and
sideboard. Match numbers and results establish the trophy; missing loss rows
remain missing. This is a source-order historical sample, not today's trophy
feed or a uniform sample of all TLA drafts. Deduplicate events, not deck content.

Native `Decklist` / `admit_decklists` resolve names (basics to `lea.*`) and
reject whole lists, including sideboards. **134 eligible / 866 rejected**;
all source lists and individual rejection diagnostics are retained. There are
134 distinct eligible normalized lists. Frozen corpus SHA256:
`2173600970d0ab9efb44c620b082e8b15a2615d4e4d801ec0c015710dc333951`.
Eligibility is never recomputed in the learner, and rejection never causes
resampling beyond the 1,000-event target. The largest exclusion counts include
Lightning Strike (237), Bender's Waterskin (225), Water Tribe Rallier (146)
and The Cave of Two Lovers (143); one list may have several issues.

## Training and allocation

The JSON regime and protocol beside this document freeze every parameter and
schedule. Each new game independently draws both seats uniformly from the 134
entries, including mirrors. Native reset seeds own these draws independently
of shuffle/action randomness. Receipts keep actual entry IDs, normalized decks,
sideboards, native seeds and Commands. No network access occurs at reset.

Use the shared MTG-181/183 recipe: 64-wide, two attention layers, four heads,
scalar masked-mean value, additive policy (FiLM off), Adam, unchanged damped
policy hyperparameters and absolute-iteration LR/tau schedule, EMA .999,
256 streams, floor 64 decisions/stream, both seats and complete games, one
producer, CUDA float32. MTG-181's three seeds improved .206 to .851 against
its old heuristic; MTG-183's longer seed improved .065 to .801 against the
stronger anchor by update 5950. These establish the recipe's evidence, not
that one seed, a later architecture or a different world is intrinsically best.

Required deviations: current world w17 and full-world vocabulary replace w13's
Elemental roster; current policy input v5 replaces historical v1. The eligible
TLA population replaces ten authored decks. Initialization is fresh, seed
21301. No historical checkpoint/world receipt is reused. The ordinary executor,
Etude checkpoint admission, attempt ledger, quarantine and 10% error budget
(window/warmup 100/100, maximum 40,000 actions/game) remain the owners.

First allocation: **one L4, at most one hour plus Etude's bounded setup window,
$1 job ceiling, one attempt**, inside a **$10 cohort ceiling**. Target 100
updates, 2,700 training watchdog seconds, 120-second stop/export reserve and
300-second finalization reserve. Export initialization, updates 25/50/75 and
actual endpoint. Etude supplies the machine deadline, timer and cleanup.
No retry or continuation is automatic. Any subsequent allocation must debit
this same pilot's retained costs before committing its cap. Unused allowance
is not a requirement to rent. Evaluation is local CPU at $0 rental cost.

Launch only from an exact committed source:

```sh
uv run etude launch experiments/tla_pilot.py --revision COMMIT --rent l4 --cohort mtg213-pilot-v1 --max-usd 10 --quote
uv run etude launch experiments/tla_pilot.py --revision COMMIT --rent l4 --cohort mtg213-pilot-v1 --max-usd 10
uv run --extra artifacts --extra evaluation etude eval experiments/tla_pilot.py --source r2://manabot/jobs
uv run --extra artifacts --extra evaluation etude report experiments/tla_pilot.py --source r2://manabot/jobs
```

## Evaluation and interpretation

Eight independent uniform corpus pairs use deck seed 21390001. Deals
21391000–21391007 are separate from collection. Four legs per pair exchange
policies and reverse deck/seat assignments, keeping both decks' sideboards.
`arena-pair-deal-player-v1` pins policy action seeds; candidate identity aliases
stay fixed across checkpoints against the same anchor. Mirrors are legitimate
if independently sampled; no holdout-deck generalization is claimed.

Every scheduled checkpoint and actual successful endpoint plays 32 games
against the fixed source-pinned heuristic. Every noninitial checkpoint also
plays 32 against this run's fixed initialization. The shared evaluator enforces
120 seconds and 10,000 Commands/game, journals pending Commands, supervises
hangs and replays traces. Etude retains games with participant `seat` and native
`game_seat`, deck IDs, seeds, scores, failures, truncations and validity. Failed
or missing legs remain in the denominator and are not retried or replaced.
Report valid W/D/L separately from attempted/expected/invalid/missing counts.
Any incomplete comparison has no qualified aggregate learning claim.

The preliminary learning criterion is endpoint heuristic gain at least +0.10,
a 95% paired bootstrap interval for the gain excluding zero, and endpoint vs
initialization 95% whole-pair bootstrap lower bound above .5. Bootstrap the
eight correlated four-leg blocks (seed 21392001, 2,000 resamples), not individual
legs. Thirty-two games have low power; an inconclusive interval is expected and
reportable. No best-checkpoint selection, post-hoc seed change or extra scoring.

Compare gains at the same collected-decision coordinates where historical
receipts support it; also report learner decisions, environment decisions,
optimizer exposures, collection/learning wall seconds, CPU/device and rental
cost separately. MTG-183's stronger-heuristic score rose .065 → .304 by update
50 and .441 by 100. Updates have variable complete-game sizes: they are not
matched decision budgets. Current rules, v5/full-catalog input, trophy selection,
anchor code and hardware differ. Neither steps/s nor a gain at 50 updates alone
establishes equal learning pace. Report pace as unresolved unless those
comparisons support it; never interpolate an unsupported historical coordinate.

A local two-stream, one-update CPU proof is implementation validation only;
it has its own receipt and does not replace the GPU pilot or score this protocol.
