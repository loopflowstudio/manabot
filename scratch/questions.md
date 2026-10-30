# Shared evaluation follow-up — 2026-10-24

Accepted by Jack Heart, 2026-10-24:

- After interruption at 40/100 complete valid deals, explicit continuation runs
  the missing 60 under one logical evaluation. Keep all attempts and count each
  block once. Fresh full repeats remain distinct. No automatic retries, native
  state recovery or mixing CPU/CUDA/numerical contracts.
- Local small-checkpoint versus heuristic checks share evaluation semantics
  without training, scheduling, remote credentials or callback transport.
- Keep compute lifetime and accounting internal; no public allocation creation
  is required. Checkpoint pairs at growing model sizes are the primary workload.
- Aim for at least two useful hours of eligible queued work before starting a
  cloud eval server, estimated for its measured host/model mix. Release idle
  compute when work finishes. This is no minimum billed lifetime. No urgent
  override or maximum queue wait has been selected.

Implementation assumptions, reversible and scoped to the supplied checkout:

- Preserve the initial attempt in the existing panel directory; subsequent
  attempts live beneath `attempts/<sequence>/`. `evaluation_status` derives
  progress from immutable block receipts and retained attempts. A UUID separates
  fresh evaluations. No second queue or progress database.
- A filesystem lock fences the logical writer. Managed RPC execution holds an
  additional attempt claim; continuation seals old attempts against delayed RPCs.
  These claims are for a shared local filesystem, not cross-host object storage.
  Moving evidence requires moving the entire evaluation directory; checkpoint
  path changes require the same digest and byte length.
- The same complete-block validator owns final statistics and continuation.
  Incomplete blocks retry whole; conflicting duplicates fail closed. Deadline
  and instrumentation may change; compute settings and scientific limits may not.
  Frozen pre-continuation results stay readable but are not silently migrated.
- Per-block replay and immutable shards are published before another block can
  erase progress. The aggregate trace remains diagnostic/calibration evidence;
  reports use row-referenced shards to avoid counting both copies.

Upstream gap rechecked on installed `etude-sdk==0.2.4`: job/cohort rental ceilings
still are not measured useful-work estimates. Etude's launch scheduler owns the
missing >=7200 useful-second admission gate for new rentals; Manabot needs
host/model workload measurements for integration. Existing idle-session release
remains correct. The design records the inspected symbols and exact gap. This is
still-required compute-policy work, separate from the continuation deliverable;
no alternate durable queue or provider helper is selected.

Unselected: actual model mix, measured current evaluation throughput, memory,
host, budget, maximum wait, latency and numerical/performance thresholds. A few
experiments per week suggests on-demand as a hypothesis, not a purchase.
Historical hardware rates, 3–5 minute startup and probe012's roughly 28 games/min
are illustrations, never runtime estimates for this policy. Whole-job cost,
CUDA parity and an end-to-end gain remain unmeasured. Broad suites belong to gate;
this implementation uses focused headless fixtures only.

The published runtime/report foundation is PR #370 at
`79506de9cae0182653e251767d35ffba5427c066`. Earlier uncommitted questions are
preserved in Git blob `d8c8916accba563add0badbfd2cf30970958ca91`, design in
`1dcb52e5bbdf3747342bc89b95d0cb17e519cd40`, and evidence in
`7bae0e735e83f08273f093dbea10422ed6f416db`. Inherited MTG-174/176/180/182
assumptions remain at `07a091bca:scratch/questions.md`; upstream MTG-183/card
history remains at `884249024:scratch/questions.md`. None selects work here.

Concurrent report demo edits and `scratch/eval-perf-demo.md` belong to the
ongoing conversation's report work and are preserved, not reimplemented here.

# Upstream historical assumptions

These inherited notes preserve context; they do not select current work.

# Inherited notes

MTG-174/MTG-176 preflight, payment-repair and recovery assumptions are retained at
`003644bda6d3ed2e1e6754186960b135ff815e76:scratch/{questions,elemental-baseline-preflight,elemental-mana-repair}.md`.
They do not select current work or authorize paid execution.

# MTG-181 assumptions (2026-10-23)

- Evaluation follows Jack Heart's 10×10 heuristic grid steer; the earlier
  MTG-174 per-pair/initialization/greedy cohorts are not run. Play/draw per game
  comes from one RNG seed (18190001), so seats are not exactly balanced
  (2,452 play / 2,548 draw); play and draw are reported separately.
- Cadence 200 updates, ten local CPU workers, chosen from a local full-grid
  calibration (~45–55 min per checkpoint). No paid evaluation.
- Budget split: three concurrent 5.0-hour L4 jobs at ≤$0.62/h ($3.10 each) plus
  guardian proofs; $0.64 left for deletion uncertainty. A failed seed is not
  replaced.
- The pods' own Etude training runs plus locally rebuilt Etude grid runs form
  the "linked Etude training report"; SDK 0.2.0 lacks the historical
  `etude.reports.Evaluation` link API used by MTG-174's isolated owner, so the
  link is the per-seed run config (job, TrainingRun) and the shared report page.
- The world moved from w10/elemental-world-v1 to w13/elemental-world-v2 since
  MTG-174; that is the re-pinned post-gate world, not a recipe choice.

# MTG-183 assumptions (2026-10-23)

- A RunPod job cannot exceed twelve hours, so "one long seed" is one cohort of
  chained `last-complete` learner-state continuations (MTG-165's $50 ladder
  pattern). Collector/RNG streams restart at each of six boundaries.
- Controller is this laptop; the Mac mini (`mini-heart.local`) did not resolve.
  Guardians bound every rental, so controller downtime only delays successors.
- MTG-182 not merged: deadline-abandoned games at each window cutoff are read as
  deadline pauses (protocol), not game errors.
- Rated spacing, references (200, 2,000), held-out skip-one pairs, 10 games per
  cell for pairs/subset grids and the saturation rule are this session's choices,
  not Jack Heart's numbers.
- Local evaluation yields while MTG-181's follower runs (Task: do not disturb it).
- The 0.03 practical saturation margin (amendment) is this session's choice.

# MTG-183 Etude continuation (2026-10-24)

- Merged #362's branch instead of rebasing (the steer said rebase): rebasing
  rewrote `8e7df92e4`, the published commit w1/w2 ran from. At landing, tag or
  otherwise keep `8e7df92e4` reachable if the PR is squashed.
- Etude's 43,000 s rental cap forces a chain of one-rental segments (amendment
  02:06 UTC), restoring the frozen w1–w7 shape. Segment hours subtract Etude's
  0.25 h setup allowance (its dollar check), not in the 19:00 formula.
- Sizing w2's rate by its stage wall seconds (103.55/h), not active seconds
  (108/h): conservative, matches the watchdog's clock.
- The evaluator reads several segment runs; `--continues` (operator-declared)
  rather than a budget rule decides whether the last run's end settles the seed,
  since the successor's run id only exists after its launch.
- Follower reads w3's checkpoints with `etude.train.read.read(source, "self-play", run)`:
  the tracked experiment is the stage id (`RunContext` name). The tracking
  storage URI is an explicit CLI argument; not verified against a live run.
- Staged-object reuse compares digests by downloading the object (simple, small files).

# MTG-183 Etude eval and R2 (2026-10-25)

- #362 and heuristic v2 are on main and merged here (merge, so `8e7df92e4`
  stays an ancestor).
- Explicit `evaluate` matches, not a `Benchmark`: the design is non-uniform
  (full vs subset grids, a pairing graph over w1/w2 checkpoints Etude never
  tracked), and `curves` cannot place non-Etude checkpoints on training axes.
- One Etude match per grid row (20 groups, ~500 or ~100 games): 10 R2 match
  deliveries per unit instead of 200; a restart loses at most the in-flight rows.
- Eval tables at `r2://manabot/evaluations/` (run folder `mtg183-eval/`): my
  choice of prefix. `state.json` (checkpoints, endpoint) stays local beside the
  run and is published with the report.
- Checkpoint agent code identity = `manabot/model`, `manabot/env`,
  `manabot/arena/match.py`, `experiments/elemental_grid.py`; editing those in
  the evaluator's pinned checkout would start new agents, so pin it.
- Stopped the old follower at 2026-10-25 00:15 UTC instead of after w2 ends:
  it was playing heuristic v1 (not the anchor) against MTG-181's live process
  and no longer admitting exports. w2's tail is admitted by a one-shot `_admit` call.

# MTG-183 launcher and w3 source (2026-10-25)

- Moved the empty schema-1 `~/.etude/launcher.db` aside rather than deleting it.
  Etude storage `r2` = `r2://manabot/jobs` (cannot be changed later): prefix `jobs`
  matches the S3 cleanup's `manabot/jobs/mtg183-*` → R2 `jobs/mtg183-*` mapping.
  On-demand `l4` boots w1/w2's image digest, not Etude's default, so w3's
  machine matches the lineage.
- The amendment says w3's source is "a published commit on main", but main
  squash-merges, so no branch commit reaches main. w3's revision is this
  branch's pushed code commit, tagged `mtg183-w3-source`; the squash lands the
  same code on main. Recorded in `elemental-183-w3.json`.

# MTG-183 sync onto #360 (2026-10-24)

- #360 split `arena/` into `sim/` and `eval/`. The branch's `legs` subset moved
  onto `manabot.eval.match.play_cell`. The evaluator's `INFERENCE_CODE` now names
  `manabot/sim/match.py` and `manabot/eval/match.py` in place of the removed
  `manabot/arena/match.py`, so its checkpoint agent identities change. The
  evaluator had played no rows, so at 2026-10-25 ~01:30 UTC its pinned checkout
  moved from `d9fc58df2` to `9871e3545` (etude-sdk 0.2.2, the merged worktree's
  `_managym` build with #360's card-capture bindings) before any row existed. The frozen
  `experiments/protocols/elemental-183*` files keep the old paths as evidence.
- `etude-sdk` follows main's 0.2.2 (the steer asked for 0.2.1, which is older than main's).

# Card evaluation (#360, landed)

The 17Lands/GIH design, investigation and implementation defaults landed with #360;
they are retained at `1e6015270:scratch/{17lands-for-bots,17lands-for-bots-evidence,questions}.md`.

# MTG-183 w3 relaunch (2026-10-25 03:51 UTC)

- Read "No segment is retried" as covering segments that learned; launches that
  never took a learning step are relaunched (amendment 03:51 UTC). The literal
  reading would have ended the seed at 2,534 on infrastructure bugs, against
  Jack Heart's direction to continue it within the cap.
- Took MTG-186's three Etude worker fixes verbatim; whichever of MTG-183/MTG-186
  lands second resolves the identical hunks.
- Deleted stuck pod `ublorrai3gb3ea` through RunPod's API: `etude terminate`
  refuses a machine with a run, and the worker would never pick up the attempt
  (Etude `Courier.bring_in` records the new inbox before fetching inputs, so a
  failed fetch never retries). Worth an Etude PR.

# MTG-183 evaluation on the Mac mini (2026-10-25)

- Segment end is the tracking bucket's `state.json` (Etude's controller DB is on
  the laptop). A segment killed without finishing tracking never ends there;
  the operator then names its successor or drops `--continues` by relaunch.
- Doppler service token on the mini (read-only, manabot/prd) is this session's
  choice for R2 keys; Etude gives jobs no storage keys. Revoke `mtg183-eval-mini`
  when evaluation ends.
- No yield to MTG-181 on the mini (Jack Heart: the laptop-only yield no longer
  applies), though MTG-181's evaluation shares the mini's 8 cores; games have a
  600 s bound and take seconds.
- The eval job's `max_attempts=1`, `max_hours=190` (to `--until`).

# MTG-183 w4 restart (2026-10-25 ~21:40 UTC)

- Never-trained launches count at the larger of the earlier provider estimate
  and Etude's receipt ($0.15 vs $0.04 for `a04845f2548b`, $0.44 vs $0.09 for
  `3828021cbc80`, whose pod was deleted outside Etude): conservative under the cap.
- A wall-bounded segment exits 0 (Etude `succeeded`) with `etude-result.json`
  at the update reached, below its `until`; its tracking closes `finished`.
- `TrainingJob.stop_reserve_seconds` defaults to 120 s or a tenth of the
  watchdog, whichever is shorter, so short-watchdog experiment jobs still admit.
- The update target uses 3,250 (w4's origin) in place of "w3's final update".
- The raw-only import accepts an `interrupted` parent only; a `failed` parent is
  refused (not needed for w3; widen deliberately if ever needed).
- Branch and tag not pushed here: the brief stops at a committed spec, and the
  operator session launches.

# Sync resolution — 2026-10-25

- Accept main's retirement of `experiments/report_prototype/`; branch report-demo edits remain in the pre-sync Git history. Runtime evaluation and performance work remains intact.
- Keep shared inference and incremental replay while applying upstream checkpoint resolution and successful-game journal cleanup in the shared supervisor.

# MTG-188 assumptions and dependencies — 2026-10-26

# MTG-206 assumptions and dependencies — 2026-10-27
# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- Superseded 2026-10-29 by Jack Heart's one-representation decision (w17, no
  JSON, tokens from `Effect`): the earlier choices to stay in w16, keep the IR's
  JSON shape and build tokens in `learning.py`.
- 2026-10-29 (sync onto T1 `01e8f9975`): T1's test layout wins; v1–v4 stay
  28 wide here and `attacking` is a v5 column (plan, item 26).
- 2026-10-29: the 14 setup-used incomplete TLA cards keep their definitions and
  are listed in one `tla::INCOMPLETE` const (key, missing behavior) beside
  `cards()`, rather than a field on every definition. Since the one-representation
  rewrite every non-token definition outside `INCOMPLETE` is eligible. Crossroads of Destiny moved to `cards/fixtures.rs` but stays
  in the default pack, because Rust integration tests and `manabot/verify` build
  games from it. `set_cards` is the set filter (pool key since renamed `"world"`).
  Undefined TLA cards are no longer listed as unsupported pool rows.
- Not rewritten (dated evidence): the Academy curated pack `local_identity` values
  keep pre-rename keys.
- Full TLA completion remains the eight-PR plan; no merge or new Task authorized.
- Settled (Jack Heart, 2026-10-30, "1a, 2a"): T3 extends the open w17 and
  policy input v5 in place; never w19/v6. Widths: player 63, card 55,
  permanent 37 plus the facts block, relation 48 (v1–v4 keep 40 / 41 / 29).
- Same merge: T2's `granted_firebending` replaces T3's `firebending`
  observation field; T2's foretold column stands for `face_down` (T3's card
  column is gone). T2's exhausted / once-each-turn counts and T3's
  `used_abilities` indices both stay: the counts are T2's, the indices say
  which ability is spent. Dropping one is a review question.
- Same merge: T2's card conversion and tests win where both converted a T2
  card; T3-only cards and tests are added. T3's refreshing `can_activate`
  helper is `offers_ability`; T2's `can_activate[_card]` stand. T2's own notes
  are at `d218e1dc2:scratch/87e012-tla-t2-casting.md` and
  `d218e1dc2:scratch/questions.md`.

Main's notes at the syncs live at `6d069d6e5:scratch/questions.md` and `origin/main@633dca159:scratch/questions.md`.

- 2026-10-29 (one-representation rewrite): five cards were filed under a later
  printing and moved to their first printing's set (see plan). Three are TLA
  reprints (Corrupt Court Official, Epic Downfall, Lightning Strike), so
  `set_cards(["tla"])` no longer lists them; a TLA-legal filter would need
  printings, not first-printing key prefixes. Followed Jack Heart's
  first-printing rule; flag for f871a8 deck validation.
- 2026-10-29: `key` and `scryfall_id` are `#[serde(skip)]` — identity metadata,
  not rules — so content and state digests stay equal to w16's and digest-bound
  fixtures keep validating.
- 2026-10-29: learning token values are stable blake3 hashes of Rust/printed
  names (no vocabulary file); the policy builds its vocabulary from observed
  (kind, value) pairs as before.
- 2026-10-29: D1/f871a8 interface change: `ContentPool.compiled_semantics`
  (IR/source hashes) is gone; manifests carry `pack_key`; rows add
  `scryfall_id`; `IrError` is `ContentError`. `content_pool`/`pool_content`/
  `Eligibility` otherwise unchanged.
- 2026-10-29 (support reorg): tests that exercise several cards stay where
  they were (Corrupt Court Official with Ozai's Cruelty in `cards/tla.rs`, Epic
  Downfall with Power Word Kill in `cards/afr.rs`); Lightning Strike has no test
  of its own, so no `ptk.rs`/`ths.rs` test files exist yet.
- 2026-10-29 (concept review): world pool key becomes `"world"` (not per-world
  `"w17"`), describing the default pack; row `semantic_key` becomes `key`.
  Coordinate with f871a8 in one comment. (The v1 coverage gate was deleted by
  Task afbd6a.)

- 2026-10-29 (assumption): superseded corpora now without consumers
  (`semantic-kernel-w15-v1`, `authored-match-w11-v1`, `authored-match-parity-w11-v1`,
  `public-commitment-parity-v1`) stay in the tree like w12–w14; archiving them
  to S3 is a separate publish step, not done here.
  Superseded 2026-10-30 by Jack Heart's steer: afbd6a archived them (and w12–w14,
  v1, learn-lesson-w4-v3) to `experiments/data/archive/conformance/`.
- 2026-10-29 (public-commitment parity, decision): bound to w17 by prefix. Only
  viewer 0's tracker fails (Command 68, South Pole Voyager revealed from library
  into hand); viewer 1 consumes the hero's Island discard (30) and
  `decline_learn` (102) over all 142 Commands. `TRACKED_COMMANDS = {0: 68,
  1: 142}`; identity-stream parity still covers the whole tape. Task 3be183
  (engine wave) owns the transport and removing the bound. No xfail.

## T3 (6875f0) assumptions — 2026-10-29

- Conditional entry reads the card owner's battlefield.
- Zookeeper's creature-type count excludes a fixed noncreature-subtype list.
- Non-+1/+1 counters live in `Permanent.counters` (serialized only when present).
- Heartless Act removes counters by a repeated resolution-time choice. Exhaust is
  per incarnation; 702.177b "haven't activated" effects are not modeled.
- Auras act on the object attached at activation (captured before costs). An Aura
  put onto the battlefield uncast goes to the graveyard (303.4g not modeled).
- Swampsnare Trap castability assumes the reduction if any legal target earns it;
  targets that would leave the cost unpayable are not offered.
- Slice 7 (Vehicles): crew offers other untapped positive-power creatures one at
  a time until covered (extra or 0-power taps omitted), then the ordinary {0}
  payment. Observed `is_animated` means "a creature only through an effect".
- Slice 8 (mana): state-dependent outputs are read when offered (no current
  cost changes them); one fixed output is still a one-option `ChooseMode`.
  White Lotus Tile with X = 0 taps for nothing.
- Slice 9: Toph's granted land type is read by predicates, targets and CDAs,
  not by observed `is_land` or LKI. Only creatures and Vehicles enter summoning
  sick (hash stability), so a Clue The Mechanist animates can attack at once.
  Flopsie's limit refuses a second blocker during declaration.
- Slice 10: gained basic land types add one mana ability choosing among their
  colors (same outputs as CR 305.6, different indices). Energybending's set is
  fixed at resolution. Ozai's replacement applies directly as pools empty while
  Ozai is on the battlefield, outside the proposed-event pipeline.
- Slice 11 (X): the X choice reuses the Modal space (`ChooseMode`, mode =
  X, labelled "X = n"). Offered X stops at the first unpayable value (no cap since the T2-sync compress; spell and activation X share `payable_x_values`). Fated
  Firepower's increase applies after prevention and doubling (the affected
  player's best order; no ordering choice is offered). Ability loss ignores
  timestamps (all keywords go, counters' included); CDAs still apply. Crashing
  Wave is deferred (spell waterbend and stun distribution).
- Slice 12: excess is measured from damage marked after replacements. Lesson
  lifelink applies to stack spells. The
  tutor offers every match plus failing to find. Hakoda's top card is shown only
  to its controller; CR 401.2's no-peeking-mid-cast limit is not modeled.
- Slice 13: granted abilities follow the grantee's own in its ability list,
  so their `ability_index` depends on what is attached; stale offers are
  rejected by the existing index checks. Activated-ability targets still
  exclude their source (pre-existing), so a Boomerang creature cannot target
  itself. Glider Staff waits for T2's airbend rather than a second one here.
- Slice 14 (Honest Work): no timestamps, so it also removes later grants and
  its "{T}: Add {C}" survives an until-EOT loss; until-EOT subtypes still add.
- Tap authority: a static haste grant changes mana output only through moves
  that already invalidate the mana cache; a grant whose condition flips without
  a move leaves the cache stale until the next invalidation, as other
  condition-dependent statics already do.
- Secret Tunnel (item 16): a departed target's LKI is found by the battlefield
  slot it occupied (`ObjectLki.permanent`) instead of capturing target
  `ObjectRef`s on the stack, as the concept review suggested; slots are never
  reused, so both name the same object and the stack shape stays unchanged.
  With more than two targets, "share a creature type" means one type common to
  all, and the first-pick partner check is pairwise (exact for two).

## T3 sync onto T2 (PR #398 head `16dfd3dbb`) — 2026-10-29

T2's design and operator choices are at `16dfd3dbb:scratch/87e012-tla-t2-casting.md`
and `16dfd3dbb:scratch/questions.md`; the ones this branch now builds on:
`CastChoices` X, `ActivationSource`, `ActivatedAbilityDefinition.effects`,
`Permanent::{activation_used, record_activation}`, `ManaUse`-keyed restricted
mana, `Game::zone_cast_cost`/`may_play_from_zone`, `ZoneEntry`.

- `with_counter` keeps T2's +1/+1 meaning (Hog-Monkey's text); `no_counters`
  reads every kind. Card-only matching rejects `with_counter`/`attacking`.
- Library-top casting is a `zone_cast_cost` case; `in_casting_zone` is gone.
- An Aura entering attached invalidates both players' mana caches.
- Crashing Wave (item 18): the stun counters are placed one at a time among
  the tapped creatures an opponent controls at each placement, and all three
  must be placed while any is eligible ("any number" of recipients, CR 601.2d
  applied at resolution). The required waterbend {X} adds to the total cost,
  not the mana value; a free cast still pays it with X = 0.
- Item 19 (with Fire Nation Palace): Name/creature types appear only while an effect sets them; spell keyword
  grants reach `Observation`, not the Academy stack view. `.pyi` now also lists `Permanent`'s
  existing `exile_if_dies_this_turn` and `temp_subtypes`.
- The intermittent "rejection must not mutate state" failures in
  `structured_offers` / `structured_attacker_offers` were the shared default
  `ContentPack`'s lazy caches in `Game`'s `Debug`; fixed at item 33.

## Item 23 (policy input v5) — 2026-10-29

- Effect-set name and creature types encode as one "characteristics set"
  flag. Their values follow from the setter (Honest Work names "Humble
  Merchant", a Citizen); the attachment join carries its definition to the
  host. No definition carries the name, so a definition reference is impossible
  without a name vocabulary in the native encoder.
- Temporary subtypes: flags for the five basic land types and Construct (every
  current grant besides v4's Avatar), then a count of any other, so none is
  dropped silently.
- `sideboard_counts` / `remaining_sideboard_counts`: the agent's remaining
  copies are already outside rows; the opponent's sideboard looks like hidden
  information that the `Observation` (and the LLM text's "Initial sideboard")
  exposes. Not encoded in the tensor. Question for Jack Heart: is a
  sideboard public in this format, or should the `Observation` hide it?
- Event `from_zone`/`to_zone`/incarnations stay out of the v0 event tensor;
  public history v1 (`policy_history_version` 1) encodes them.
- Item 24 adds the state the `Observation` lacked (combat assignments, lost
  abilities, spent activation limits, kept red mana, no maximum hand size).
  Spent limits encode as flags for ability indices 0–3 plus a count of later
  indices (no current card has more). `deathtouch_damage` and `was_cast` stay
  unobserved: SBAs consume the first before any decision, and the second is
  read only while its own entry resolves.

## T2 notes at the second and third merges (heads `89e320c48`, `a5587df5a`) — 2026-10-29

T2's operator choices are at `a5587df5a:scratch/questions.md`. New since
`d218e1dc2`: possible worlds v3 resample foretold cards (ordered slots,
`P(m, F)` fill factor); belief priors are not yet foretold-aware.

- 2026-10-30 (continuous effects as records), implement-time refinements of the
  accepted model, reversible:
  - A set fixed at resolution (CR 611.2c) is one record per object sharing a
    timestamp, rather than a `Locked(Vec<ObjectRef>)` record, so every read
    indexes by affected object.
  - Durations are `EndOfTurn` and `Indefinite` only; `EndOfCombat`,
    `UntilYourNextTurn` and `WhileSourceOnBattlefield` join when a card needs
    one (no T3 card does). "Grant ability" stays the Equipment/Aura static.
  - The timestamp clock is the object-id generator (a permanent's `id` is its
    entering timestamp; attaching renews `Permanent.timestamp`).
  - Rules effects (can't be blocked, exile if it would die, kept mana, hand
    size) are not abilities, so losing all abilities does not end them.
  - The Observation keeps `plus1_counters` (pre-T3 consumers) and `counters`
    holds the other kinds, so they partition rather than duplicate.
  - The model hashes fact ids into 4096 embedding buckets (collisions
    possible); reusing the semantic-card token table for fact ids is a
    follow-up if it matters.
  - Ability descriptions of continuous effects are generated uniformly
    (`continuous_description`), so some offer texts change wording
    ("can't be blocked until end of turn").

## T2 and afbd6a notes from main (merged 2026-10-30)

- Printed flashback plus Iroh's grant: the grant's cost is used.
- Synced onto T1's w17 (`8790821da`): T2's 51 cards were regenerated in engine
  types by T1's generator (adapted for X costs and restricted mana) and proven
  Debug-identical to T2's prior lowering for all 206 definitions; T2's
  authoring ops, IR opcodes 66–79, schema-8 growth and Python token-builder
  edits are gone with the layer. T2 extends the open world **w17** in place
  (Jack Heart, 2026-10-30; it was w18 until then).
- Card abilities use `ActivateCardAbility` (sharing `PriorityActivateAbility`);
  a discard-cost ability refers to its card's graveyard object after costs.
- `lesson_mana` stays the total of any restriction; per-class detail is
  `restricted_mana`. `PlayerVitals` ≤ 40 bytes, `UndoEntry` ≤ 72.
- Avatar's Wrath's hand-only lock ends at its caster's next untap and blocks
  flashback too; mass airbend exiles one at a time.
- Kicked Aang's Journey runs two searches (no TLA Shrine is defined yet).
  Dai Li's revealed hand is public `known_hand`. Elemental Teachings with fewer
  than two lands: the opponent bins all.
- X is offered as `ChooseMode { mode: X }`, 0 through the largest payable
  value. Spirit Water Revival's "waterbend {6}" is a waterbend kicker.
- Free casts during resolution (Yue, Solstice) are a nested
  `Decision::CastWithoutPaying`; valid because the free cast is each card's
  last instruction.
- Hama/Azula: `Effect::ExileLinked` creates the source-linked grant; "any type
  of mana" prices the cast as generic equal to mana value.
- Foretell is `PriorityActivateAbility` (Academy labels it "Foretell X"); a
  face-down move is hidden from history and events for both players.
- v5 tensor: restricted mana per spending subtype × color (a multi-subtype
  class counts under each); announced X on the stack spell's card row; a grant
  whose source has left is not shown. The w17 guard surfaced four T2 status
  fields with no tensor column; v5 now carries them (permanent 29 → 32: used
  exhaust abilities, used once-each-turn abilities, granted firebending;
  player 59 → 60: no maximum hand size), with JSON, text and Academy types.
- T2's `mod casting` tests were merged into T1's flat `tests/cards/tla.rs` in
  card order; T2-only primitives joined `support/` (`cast_card`,
  `activate_card`, `move_from_library`, `can_activate[_card]`,
  `offered_targets`, `offered_cards`, `zone_of`), each defined once.
- Gaps: granted land plays of another
  player's card would enter under its owner (unreachable: no T2 grant lets a player play another player's land). (Academy's card face now prints
  and names exhausted / once-each-turn / granted firebending, text-5 wording.)
- Possible worlds v3 (2026-10-29): foretold slots are ordered (each has its
  own foretell turn), so the fill factor is `P(m, F)`, not `C(m, F)`; the
  ratio is constant across worlds. Materialization picks the slot set
  uniformly but keeps an already-foretold chosen card in place, so slot
  assignment is not uniform over orderings. Not yet foretold-aware (follow-up
  for belief owners): `PhysicalHandSampler`'s hypergeometric prior and
  `BeliefState.draw_unknown`/hidden-draw convolution in `belief/range.py`
  still treat every pool card as hand-or-library, as does the library
  count row in `belief/encoding.py` (pool − hand).
  2026-10-30 (concept review 4, 1b): the tracker's accounting now counts the
  slot; which hand card was foretold is assumed uniform by count (MTG-215).

## afbd6a (delete old world), 2026-10-30 — headless choices

- Current-world checks in `test_temur/test_sacrifice/test_elemental/test_optional_payments`
  stay in those files (they already run on `setups.json`); only compiler assertions
  were deleted.
- Phase-matrix rule-family closure against `v1/coverage.evidence.json` dropped; the
  matrix keeps its other structural checks. `matchup_source_digest` left receipts;
  conformance decks come from `cards::setups`.
- `TRAILING_ORDER` removed with `LEGACY_ORDER`: default pack is name order.
- Statics live in `CardDefinition.statics: Vec<StaticAbility>` (not `Ability::Static`)
  so trigger indices into `abilities` are unchanged. Ward is listed as a static per
  Jack Heart's direction though CR 702.21 makes it a triggered keyword; registration
  still expands it to its trigger. `blocking_restriction`/`block_restriction` merged
  as `StaticAbility::BlockRestriction(CantBlock | CantBeBlockedBy)`.
- Academy curated pack manifests keep `source_uri` provenance strings naming
  `content/semantic/v1/...` and `cardsets/tla.rs` (frozen pack assets; not rewritten).
- Token first printings (Scryfall oracle-id earliest print, 2026-10-30): Ballistic
  Boulder tltr, Bear tktk, Clue tsoi, Food teld, Monk tdmu, Human Soldier tsoi,
  Treasure txln; Ally, Dragon, Soldier, Spirit first printed in ttla.

## afbd6a sync onto T2 (c7979d310), 2026-10-30

- T2's new `CardDefinition` fields became variants: `StaticAbility::{Flashback,
  Foretell, FlashIf, CostsLessPerGraveyardCard}` (the old
  `cost_reduction_in_graveyard` flag on `cost_reduction_per` is its own
  variant) and `AdditionalCost::Waterbend` (was `kicker` + `kicker_waterbend`).
  `kicker()` returns either optional cost, so the kicker card column and the
  kicked cast stay as T2 built them. Each new variant has a guard row, witness,
  token test and real-card test in `observability.rs`.
- Fire Nation Attacks now produces Soldier, so it left `TOKENS_AWAITING_PRODUCER`.
- AGENTS.md T2 note on `Cost.zone`/`Program::landcycling` named the removed IR;
  rewritten as `ActivatedAbilityDefinition.zone` (`ActivationZone`).
- `manabot/sim/heuristic.py` reads T2's activated-ability `effects` list and
  skips hand/graveyard abilities (landcycling, foretell) when picking prompt,
  spare-mana and pump abilities. T2's nine new targeted spells are pinned in
  `test_heuristic.py`; the seven it cannot rate (airbend, linked exile, …) get
  polarity `None`, the documented neutral fallback.

## T3 sync onto main `10eeb1f55` (afbd6a: statics and costs as variants), 2026-10-30

- T3's new `CardDefinition` fields became variants, never fields:
  `StaticAbility::{LookAtLibraryTop, CastFromLibraryTop, AttachedHas,
  AttachedSetsCharacteristics, CostsLessIfTargets, KeywordGrant, Restriction,
  CreatureWithCounters, TypeGrant, AttacksDespiteDefender}` and
  `AdditionalCost::RequiredWaterbend` (Crashing Wave). Accessors on
  `CardDefinition` read them; `AbilityGrant::index` indexes `attached_grants()`
  (the `AttachedHas` variants in printed order). Each has a guard row, witness
  card and behavior test in `observability.rs`.
- Ability loss still gates main's `BlockRestriction` / `CantBeBlockedIf`
  statics (T3's `has_lost_abilities` checks kept).
- T3's Dragon and Monk definitions left `tla.rs` for main's `tokens.rs`
  (main's Scryfall ids); Monk's prowess uses T3's continuous-effect record.
- `manabot/sim/heuristic.py` reads `ApplyContinuous` as the targeted effect it
  amounts to (`BuffTarget`, `GrantKeywordsToTarget`, …; a source P/T change is
  `ModifyUntilEot`). T3's eight new targeted spells are pinned in
  `test_heuristic.py`; seven are unrated (polarity `None`).
- w17 evidence re-recorded in place (semantic kernel 14 cases / 3401
  commands; authored-match receipt and parity 142 commands; public commitment
  69 commitments, no gaps; Academy demos, choice navigation, trigger order).
- Realign repaired the stale `ModifyTriggeringPermanent` audit anchors (rules
  400.7j, 608.2k now anchor `AffectedSpec::TriggeringPermanent`).
