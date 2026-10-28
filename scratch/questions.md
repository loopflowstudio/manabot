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
