# Questions and assumptions

Inherited Wave rename, decklist, TLA, evaluation and characteristics notes are
historical context at `460732f81:scratch/questions.md`; they select no MTG-237
work. Main's newer decklist decisions and the current Task decisions follow.

## T5 historical questions and assumptions

The following T5 notes and its sync record are inherited evidence from main
`a8abf01b1`, not new directions or current validation for MTG-237.

Accepted decisions and pending checks live in
[jack-heart/a7234f-tla-t5-permanent.md](jack-heart/a7234f-tla-t5-permanent.md).
Historical parent notes (D1, T2–T4, archive/CI work and MTG-214's Wave rename)
are evidence, not directions for T5. Complete pre-compression notes:
`bbb5101eb:scratch/questions.md`; older stacked notes:
`1e9fd2d7c:scratch/questions.md` (lines 5–472).

### T5 implementation assumptions

- Surveil reuses LookAndSelect: select moves the inspected card to the
  graveyard, decline keeps it. Kept cards retain order; exact for surveil 1,
  the only amount authored. Supporting larger amounts needs ordering choices.
- Copy exceptions share `Modification` with continuous records. The protocol
  calls the field `copy_of` because Pydantic's `BaseModel.copy` occupies `copy`.
- Airship uses self-relative `CreatureWithNamesakes` because
  `check_static_condition` has no source. It reads private layer stages,
  never recursively reads `characteristics()` during layer computation.
- T4 owns `PlayerTurnHistory`, sacrifice bookkeeping and its column after
  attacked. Text uses singular/plural correctly; no retained fixture holds
  the old “1 permanents” / “1 cards” wording.
- The engine measured 314 supported definitions (257 TLA eligible plus 57
  others; four TLA and one ths unsupported). Earlier 239/266 notes were
  pre-sync or based on stale main counts, not current measurements.
- The permanent row's name/creature-types fields are Aura-set markers; the
  card row states layered characteristics. Ward is printed unless abilities
  are lost; no authored effect grants ward. Text and mana cost stay printed
  until a modification changes them, then join `Characteristics`.

### T5 reversible choice awaiting a separate product decision

Since 1a/2a the projection's `base_power` / `base_toughness` contain layered
card-row P/T on the battlefield. No UI component reads them for printed P/T,
and no other Python-visible source exposes printed P/T. Keep and document
that value for T5; a printed-P/T accessor or field removal requires a follow-up
protocol decision by Jack Heart. This is not a T5 blocker.

The choice-navigation re-record (`889948346`) preserves recorded offer ids
when labels repeat, keeping original state digests and offer order. It is
published; Academy UI passed at `bbb5101eb` (CI 41539137476). Gate subsequently
confirmed all required CI at `1de85f7aa` (run 41541743395), plus local copy-fact
parity coverage and its CI selection. The latter remain uncommitted; see the
design's gate results for commands and evidence boundaries.
Shipping remains stopped by steer f39fef34; `/verify` is not a pre-merge check
because that skill finishes delivered Tasks.

## MTG-216 (decklist disclosure) — 2026-10-30, implementation (headless)

Jack Heart accepted one open/closed option, default open (supersedes the
kickoff's four-way assumption). Reversible choices made while implementing:

- No player-row "known" flag: an open main deck is never empty, so all-zero
  `decklists` rows mean closed. The player width is untouched (67 after the
  sync with main's T4 v5 widening) and the native width-binding rule is
  unchanged.
- No separate Python `DecklistHidden` class: refusals are `AgentError`
  ("decklist closed: …"). `Game::determinize` and
  `HiddenHandConstraints::for_viewer` assert; `Env` entry points check first.
- Search refusal at match build lists observation-only kinds
  (`checkpoint`, `heuristic`, `random`, `scripted_greedy`); every other kind is
  refused in closed matches. Academy refuses its search villain in closed games.
- Decklist pooling is semantic-pack only, like `known_hand`; generic-model
  policies ignore the `decklists` buffer. Adding `decklist_projection` changes
  the v5 semantic `ActorCritic` parameters, so v5 semantic checkpoints saved
  before this change fail strict load (open world; none frozen).
- Disclosure is not part of `world_binding` or deck identity: a checkpoint can
  play both modes, and in a closed game it sees zero opponent rows.
- `Deck.player()` (training deck pools) seats an open decklist; it takes no
  disclosure argument. Closed games come from `MatchConfig.decklists` or
  Academy only.
- `canonical_replay`/`study` privacy validators do not check decklist maps:
  frames carry no disclosure, and the engine projection is the authority.

Inherited evaluation, MTG-17x/18x and card-evaluation notes (none select T4
work) are at `ee42a8685:scratch/questions.md`, above "TLA direct-Rust decisions".

## MTG-237 (engine budget), 2026-11-02 — headless choices

- The headless steps/s workload is native (`Env::step`, both seats uniformly
  random over legal actions, 512 seeded games, skip_trivial on), not the
  rul-9 Python semantic-Command path (its runner was deleted in #343). It
  measures the engine the training environment steps, builds in the same
  cargo invocation as `bench_search` and needs no extension build in the job.
  Its steps/s is therefore not comparable with rul-9's 588.5 (Python,
  semantic Commands); the baseline is its own.
- Digests are asserted identical for both workloads, so any rules change
  reaching the curated decks (or the deal) moves the baseline deliberately
  with a reason, as the review asked for search results. The random-play
  digest also sees every legal-action count; search now covers all ordered
  scores, not only winning actions. Expected cost: one `record` per rules PR touching these cards.
- The budget runs as a `ci.yml` job via `workflow_call` so the Tests Result
  aggregate owns it and docs-only PRs skip it; every job is its own VM, which
  is the quiet runner. `ubuntu-24.04` is pinned rather than `ubuntu-latest` so
  the baseline's hardware profile does not move with the label.
- Counterexample to the design's committed absolute numbers (2026-11-02): the
  four dispatches landed on three CPU models (EPYC 7763, Xeon 6973P-C, EPYC
  9V74) and the identical T4 engine spread 6.8k-10.3k steps/s between them,
  far outside 10%. The comparison is therefore A/B on one machine: the job
  rebuilds the reference commit with the change's benchmarks and compares
  the candidate to that fresh run; the file holds the reference commit,
  digests and its recorded figures. Two EPYC 7763 runs of one engine agreed
  within 2.5%, so 10% stands for now. Jack Heart's review decides whether to
  keep this reading of "committed baseline".
- T6's head (`2c1ba169f`) is unpushed, so measuring it on CI needs its commit
  reachable from origin; pushed as a plain ref under `refs/engine-budget/`
  for the dispatch and deleted afterwards (not a branch, no PR).
- Headless readings (2026-11-02; decisions, review findings and the
  proposed Wave memory are in the plan): the two formerly draft decisions
  (A/B against a committed reference commit; native `Env::step` random play)
  are read as implied by the accepted outcome plus the runner evidence, so no
  review returned Blocked; Jack Heart's PR review confirms or reverses them.
  The artifact is named `engine-budget` in both modes (the run id identifies
  a measurement); `measurement.json` carries the full `lscpu` model name,
  orientation only; compare's FAIL hint leaves `<run-id>` as a placeholder so
  the line reads the same in CI and in tests; the artifact also contains the
  `reference/` run, which `record` ignores; `record` takes the tolerance from
  the committed file rather than defaulting it. Running the candidate before
  the reference on the one VM is assumed not to bias the comparison beyond
  the 10% tolerance (two runs of one engine on separate EPYC 7763 machines
  agreed within 2.5%); the order exists so a run whose reference cannot build
  with the change's benchmarks still records its candidate. Reverse it only
  with evidence that order matters.
- Fourth headless concept review at 078d37ef4: interpret the Task's "score
  digests" literally as preserving the ordered action scores, not just the
  winning action. This follows the accepted correctness requirement; it is
  not new approval from Jack Heart. The counterexample and next implement
  direction are in the working design. Keep all completed workflow work and
  historical measurements; a new digest needs a new recording, not rewritten
  historical evidence. No open product choice; no shipping approval.

## T5 historical sync onto main 230b2f563 — 2026-11-02

The design link above is absent in this checkout (removed with added scratch
files at 2c6cdae5f); the retained assumptions supply the resolution context.
Preserve T5's CopyView and layered battlefield P/T alongside main's decklist
disclosure fields. Navigation prefixes, offer IDs and state digests stay fixed;
recompute derived frame hashes for the combined projections. Reconcile the
current authored-match parity receipt's source manifest with the merged source,
then verify the unchanged command tape and behavioral evidence.

The first focused run passed all navigation cases but found stale parity viewer
hashes. The producer confirms that only `viewer_observation_sha256` and
`viewer_state_sha256` differ: combined layered P/T and decklist projections
require new hashes. State witnesses, command tape, transitions, presentation,
and stale-object proof remain identical. Refreshed those derived hashes from
the producer; no behavioral assertion was relaxed.

Focused verification: navigation plus authored-match parity initially reported
9 passed / 1 stale receipt; after correcting only the derived viewer hashes,
the failed parity test passed (1 passed). The advice schema equals its Pydantic
producer. Native extension rebuilt and replaced atomically. No push or landing.

## MTG-237 sync onto main a8abf01b1 — 2026-11-02

Resolved only the notes conflict by retaining both independent histories.
MTG-237 decisions and completed implementation remain intact; inherited T5
notes select no additional work. The waiting sync caller owns verification
and publication after this recovery exits. No shipping approval.
