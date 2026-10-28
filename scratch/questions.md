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
- `.lf/wave/engine/{GOAL,MEMORY}.md` are absent. The accepted Task handoff supplies
  direction; no Wave memory is modified.
- MTG-177 still reports only merged PR354 as delivered. Its later resolution
  repair is not integrated or presumed accepted. Independent content work
  proceeds; final fidelity claims must name that remaining dependency.
