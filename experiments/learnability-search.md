# Broader learnability search — 2026-10-15

**Completed:** the 30-minute primary run and two concurrent follow-ups found no new
production defect. Closest-pair sampling increased definition-only coverage from
4 to 93 pairs on the same 352 states. See [results](#results--completed-2026-10-15);
the preregistered questions and instrument corrections below remain as history.

## Frozen first-run question

Does a 30-minute CPU exploration expose action information loss or pipeline
invariance failures beyond the fixed PR suite? Jack Heart authorized this
prototype and a roughly 30-minute isolated remote run, with spending below $5.
This public repository uses a standard GitHub-hosted Ubuntu runner: no purchased
compute, credentials, laptop training, mini jobs, or scheduled nightly enabled
by this prototype. The existing short diagnostic's schedule is a separate
pending cadence decision.

Prediction, before execution: at least six distinct native decision families
will be audited, at least 100 distinct states will enter completed shared fits,
and at least 90% of per-case conditional fit scores will exceed 0.95. Predict
zero confirmed encoder/reordering/reload defects. Failure to fit is an
investigation candidate, not proof of missing information or a CI failure.

The runner generates eight seeded authored-game prefixes per round (up to 1,200
native steps each), swaps the two deck assignments, and samples up to six states
per decision family. At most 32 states enter a round's shared models. Native
versus scalar encoding and reversed-native-offer parity are audited before
sampling. Exact game seed, deck assignment, action prefix, state digest, raw
observation and encoded tensors are retained for every selected state.

Public action descriptors include verb, declaration, mode/ability selectors,
focus roles, definition, visible permanent attributes and keywords. Instance IDs
are excluded. Equal descriptor copies share mass; lexicographic first/last
classes define opposite artificial preferences in two separate shared fits.
The loss measures probability conditional on those two classes, not the chance
of selecting that action among every legal offer. No strategic label or
held-out generalization claim is made.

Three width-32 configurations: generic with attention, semantic with attention,
and generic without attention. Each uses paired initialization seed 1000+round,
240 Adam updates at .003, and P>=.95. Frozen-policy assertions check batching,
offer/object ordering, legal masks and state-dict reload at tolerance 1e-4.
The structural assertions inspect ordinary policy probabilities; reload is a
diagnostic weights round trip, not a world-bound playable checkpoint claim.

Cost/stopping: one standard CPU runner, one Torch thread, 1,800-second total
search budget checked in generation and each update; external 1,830-second
process watchdog, 40-minute job limit including setup. Save each completed fit;
record a final incomplete fit explicitly. No retries to obtain passing results.
Errors remain in artifacts. Separate native audit violations from optimization
misses. This is exploration, not a blocking PR gate or strength campaign.

Strongest confounds: public descriptors may omit relational distinctions; random
play may never visit rare decisions; small shared models and arbitrary ordering
labels may fail from optimization/capacity rather than aliasing. Diagnosis must
replay exact failures, check information flow, and distinguish those explanations.
An unchanged source rerun is not independent evidence of a repaired defect.

Execution entry point:

```sh
<<<<<<< HEAD
uv run experiments/runners/learnability_search.py --out .runs/learnability-search --seconds 1800
=======
uv run python -m manabot.verify.learnability.search --out .runs/learnability-search --seconds 1800
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
```

The temporary branch-only push trigger requires `[learnability-search]` in the
commit message. Normal PR updates do not launch another 30-minute experiment.

## Instrument correction, while first run continues

Run 38782530929 at source 6c445836 started the 30-minute search. The live log
shows completed generic/semantic fits, but immediate configuration errors for
the attention-disabled arm. Inspection reproduced the validator rejection:
`attention_feedforward_dim=64` is forbidden when attention is disabled. This is
a prototype configuration error, not a learned-policy defect. The initial run
and its errors are retained and continue to supply the other two arms.

Correct the attention-disabled setting to `None` and launch a separate
300-second supplemental run for that arm alone. Its first rounds reuse the
same generation/model seeds and fixed update budgets, permitting matched-case
comparison. It runs concurrently on a second standard GitHub runner ($0),
without extending the primary 30-minute run or changing its inputs. No rerun
of successful arms or thresholds is selected. The branch push trigger for this
revision selects only the supplemental arm; manual dispatch retains explicit
budget/configuration inputs. Both run receipts will remain in the final report.

## Focused pair selection — declared before the supplemental run

The completed 300-second correction run retained 736 states (704 scored),
with no native audit or completed-fit failures. Post-hoc input analysis found
that erasing declaration creates exact input collisions in 92 retained states;
erasing definition joins creates them in 152. Clean inputs had none under this
fingerprint. These are ablations, not historical runtimes or newly found bugs.
Only 11 of the originally selected pairs isolate definition identity alone;
the extreme-class sampler often chooses an easier verb distinction instead.

While the unchanged primary run finishes, launch one additional 600-second
standard-hosted CPU search with generic and semantic attention configurations.
Keep generation seeds, budgets, thresholds and retention unchanged; change only
pair selection to prefer equal verbs and then the fewest differing public
fields. Definition keys count as categorical differences, not numeric distance.
Lexicographic order resolves ties and preference direction. This is a new,
explicitly exploratory condition; do not pool its scores with the primary run.
Prediction: at least 50% of sampled pairs share a verb and at least 90% of
completed fit scores exceed .95; any misses require replay/diagnosis.
This concurrent run stays within the original wall-time window and $0 compute.

## Results — completed 2026-10-15

No new production representation or learning-pipeline defect was found in these
runs. Every completed fit reached the declared threshold on every selected case,
and all measured batching, offer/object-order, mask and reload invariants passed.
This does not certify strategy, generalization, every action pair, or every state.
Each shared model fits at most 32 fixed states, not the entire accumulated corpus.

| Condition | Search wall time | Completed fits | Distinct states scored | Per-case preference scores |
|---|---:|---:|---:|---:|
| [Primary generic + semantic](https://github.com/loopflowstudio/manabot/actions/runs/38782530929) | 1,800.11s | 107 | 864 | 3,424 |
| [Corrected attention-disabled control](https://github.com/loopflowstudio/manabot/actions/runs/38783322327) | 300.03s | 44 | 704 | 1,408 |
| [Closest-pair generic + semantic](https://github.com/loopflowstudio/manabot/actions/runs/38784811065) | 600.03s | 43 | 352 | 1,376 |

State sets overlap; do not add the distinct-state column. Each run retained one
unfinished fit at its deadline, excluded from completed scores. The first run
also retained 54 immediate configuration rejections for the attention-disabled
arm. Its feedforward setting was invalid; the supplemental run repaired it,
and a cheap preflight regression now constructs and scores all three
configurations. These errors were in the prototype, not the production learner.
GitHub job success means execution completed, not that every attempted arm passed.

All 6,208 completed per-case scores passed; no initialization score reached .95.
All runs audited ten native families: priority, attack, block, target, scry,
look/select, optional payment, Learn, waterbend, and cleanup discard. The primary
run completed 185 games and capped 31 game prefixes at 1,200 steps. It audited
30,695 multi-class decisions before sampling the 864 retained states. Modal and
legend-rule decisions were not observed; the existing modal fixture remains
necessary. The control's retained boards reached 39 permanents, 73 visible card
rows (including outside cards), and 18 legal actions; this is still below the
full representational capacity and does not test every overflow boundary.

### What improved the instrument

On the exact same 352 native states, the original sampler selected only **4**
definition-only pairs; the closest-pair sampler selected **93**. Their state
digests and action traces match. The latter selected equal-verb pairs in 227/352
states (64.5%), satisfying its declared coverage prediction. All its completed
scores passed; the minimum conditional probability was 0.9999803. This is
capacity on artificial preferences, not evidence that identity is strategically
relevant in every pair or that the model will choose well during ordinary play.

An independent post-hoc audit checks every offered action's encoded row and
focused object inputs, including definition joins, against the raw public
descriptors. In the primary corpus it found **0 clean input collisions**.
Erasing attack declaration recreated collisions in **108 states**; erasing
definition joins recreated collisions in **179 states** (205 collision classes).
These are controlled ablations, not new bugs, historical runtime executions, or
a proof that the learned encoder is injective. They demonstrate that the
retained states expose the information-loss mechanisms behind the previous bugs.

One saved state per observed family was replayed from its native seed and action
trace. Native state digests and all encoded tensors reproduced exactly, including
Linux-generated evidence checked on macOS. A separate lightweight ten-state
replay took 0.303s with no model construction or training.

### Cost, receipts, and disposition

All training ran on standard GitHub-hosted Ubuntu runners in this public
repository: **$0 purchased compute**, no launcher credentials, and no laptop or
mini training. The two supplemental runs overlapped the primary, so the remote
work occupied roughly 31 minutes of elapsed time including primary setup, with
45 minutes of search CPU allocation plus setup across runners. Actual recorded
search CPU totals were approximately 44.99 minutes. No existing experiment was
stopped or changed.

Compact receipts include exact source commits, original result hashes, Actions
artifact IDs/digests, all configuration errors, per-family counts, representative
scores, replay checks and collision controls:

- [Primary receipt](../docs/evidence/learnability/search-primary.json)
- [Corrected control receipt](../docs/evidence/learnability/search-control.json)
- [Focused receipt](../docs/evidence/learnability/search-focused.json)

Full reports, model weights, traces and tensors are retained in the three Actions
artifacts for 14 days and locally under `.runs/learnability-search-evidence/`.
Reproduce the all-offer input audit and one native replay per family with:

```sh
<<<<<<< HEAD
uv run experiments/runners/learnability_input_audit.py .runs/learnability-search-evidence/primary --replay
=======
uv run python -m manabot.verify.learnability.input_audit .runs/learnability-search-evidence/primary --replay
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
```

The experiment supports spending extra compute on state/choice diversity and
configuration coverage, not extending already successful fits. Pair selection
was more consequential than the longer budget: target near-identical choices,
keep exact replay evidence, and retain the fast fixed regressions. A nightly
schedule is not selected by this result. The ordinary suite now runs on relevant
PR updates and manual dispatch; the broader workflow is manual only, with
explicit configuration, pairing and budget inputs. Temporary branch push
triggers were removed. Further exploration should target omitted families and
single-field distinctions before expanding runtime alone.
