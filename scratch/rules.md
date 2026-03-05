# Rules Wave — Scratch Notes

Design decisions and architectural context are captured in the wave itself:
- `wave/rules/README.md` — vision, goals, risks, metrics
- `wave/rules/01-12` — stage specs
- `scratch/jack-heart.mtr.20260304_2103.md` — full design conversation record

Key architecture decisions (also in the wave stages):
- Declarative effect DSL, dual-purpose: executable + observation-encodable
- Sequential sub-decisions for targeting (like attackers/blockers)
- Engine-side target legality filtering
- Event system as foundation for triggers/replacement/continuous effects
- Rust-first rule harness now (`tests/rules/helpers.rs`), JSON trace frontend deferred
- Training smoke tests woven through every stage
