# TLA content decisions — 2026-10-28

Jack Heart accepted seven TLA PRs plus separate decklist Task f871a8. This Task
implements only the first content cut. The earlier MTG-207 assumptions are
retained at `c5d406e3:scratch/questions.md` and do not direct this Task.

- Current rules come from `content/semantic/current/definitions.json`; setup
  projections select its transitive closure. Frozen v1/w11 documents remain
  evidence. Current presets retain their names and deck/sideboard counts while
  receiving new compiled hashes and world w16 (w15 reserved by MTG-177).
- Pool eligibility is distinct from existing authored setup availability. A
  preserved program is not a whole-card fidelity certificate. Only explicitly
  qualified definitions can enter the decklist pool; unsupported inventory is
  reported separately. Task f871a8 owns admission and parsing.
- The final research handoff is preserved under `reference/full-tla-final/`.
  Ten raw source hashes and 72 repository source hashes match the pins/base.
  Its allocation supersedes the early sketch; five demonstrated prerequisites
  are reassigned in durable `docs/rules/tla/allocation.jsonl`.
- `.lf/wave/engine/{GOAL,MEMORY}.md` are absent. Legacy `wave/rules/GOAL.md`
  and `MEMORY.md` were read; there are no child MEMORY.md directories there.
  Their selected-matchup scope and old current-world claims are historical;
  Jack Heart's explicit 2026-10-28 TLA direction governs this Task. No alternate
  Wave was selected or memory modified. Proposed engine-memory lesson: authored
  setup availability, independently qualified pool entry and full-file completion
  are separate claims; closure hashes prevent stale evidence, not missing proof.
- MTG-177 now reports gate running, with only earlier PR354 retained as delivered.
  Its later resolution
  repair is not integrated or presumed accepted. Independent content work
  proceeds; final fidelity claims must name that remaining dependency.
- Reversible allocation decision for this pass: retain all 91 T1 owners while
  identifying existing T2/T3/MTG-177 prerequisite owners per card. Of the 60
  reviewed holdouts, 54 now qualify and six stay unsupported; neither copying
  shared repairs nor adding cyclic delivery dependencies is authorized by this
  disposition. Full-card qualification remains with T1 after accepted integration.
- Haste grants must affect attack and tap-symbol legality without clearing
  summoning sickness. This small established-keyword repair fits T1's Wartime
  Protestors review; broader effective-characteristic work remains T3.


## Historical notes retained from the merged main branch

# Rules audit assumptions — 2026-10-26

- The five-area audit is complete at its recorded evidence depth; the accepted
  2026-10-26 resolution repair is integrated with MTG-195 for review. Unsupported mechanics
  can be marked missing only after inspecting the relevant representation and
  authority; an untested or unresolved obligation stays unknown or partial.
- Directed synthetic fixtures expose precise obligations, not proof that the
  selected authored decks can reach them. No new Task or identity migration is
  inferred from a synthetic probe alone.
- MTG-195 owns trigger ordering; MTG-196 owns replacement ordering. Their
  mechanisms remain outside this repair; no external messages are selected.
- Inherited MTG-184 questions are preserved at
  `324db65f2:scratch/questions.md`; they do not direct this audit.
- Resolution repair (2026-10-26): use clause-specific eligibility slots in the
  existing EffectFrame; preserve invalid positions rather than compacting targets.
  Untargeted trigger context is used only when no targets were chosen. Existing
  stack CardId identity remains unchanged; no broader stack ABI is inferred.
- Integration (2026-10-27): Jack Heart requested stacking on MTG-195 and retried
  after its delivery. PR386 has merged, so sync main incorporates it directly.
  Preserve trigger-ordering context/action joins and resolution eligibility together.
- Combined contracts: world w15, hash 10, policy-input and capture/text v4,
  embedded decision schema 7. V2/v3 keep 32 relation columns; v4 adds eligibility
  at column 32. Validity remains column 31. Parent w14/v3 evidence stays frozen.
  Fresh receipts are recorded; broader world admission remains gate-owned.

- Final identity (2026-10-27 implementation): use w15/hash 10 and policy/capture
  v4 for the combined source. Re-record the existing bounded w14 inputs at new
  paths; keep all historical files. Broader acceptance stays with gate.
- Line accounting measures integration-authored differences against both merge
  parents and w14 fixture inputs separately from inherited bytes. Report full
  new runtime-fixture line counts as well; do not hide any ceiling conflict.
  Current churn and overage are recorded in `line-accounting.json`; retain the
  coherent world/fixture cut and disclose the overage rather than omit consumers.

## Preserved main-branch notes (historical context)

# MTG-207 assumptions — 2026-10-27

Inherited MTG-206 notes are at `4ab16a140:scratch/questions.md`.

- **Design chosen without review.** The Task left design open; headless, the
  simplest exact path was taken: batch `estimate` by deal count and keep the
  page one self-contained file computing every combination. Per-pair payloads
  were not adopted: without a server, every combination must still be
  computed when the page is written, so splitting saves browser memory, not
  render time.
- **254 MB page.** The whole-experiment JSON is 254.2 MB at base and after
  (unchanged bytes of values). Whether a browser should load that is a page
  design question left for review; values were not rounded, since the
  acceptance compares them exactly.
- **etude-sdk 0.2.11** (etude #72) merged 2026-10-28 02:09Z but was not on
  PyPI or GitHub releases at 03:24Z (still 0.2.10 at 03:40Z), so the pin stays 0.2.10 (confirmed by Jack Heart's operator steer 2026-10-28). Projection with
  Etude main's `estimate` swapped in is recorded in `scratch/mtg207.md`.
- **Machine load.** Load average 90–240 on 16 cores throughout; wall times
  are inflated several-fold, CPU less so.

- Shipping gate (2026-10-28): use CI's debug optimization level 1 with two
  jobs/test threads. The first unoptimized full run was interrupted after its
  search-state contract binary exceeded five minutes; no assertion was relaxed.
  Both the incomplete run and the complete acceptance result belong in durable
  evidence. No additional randomized exploration is selected.
