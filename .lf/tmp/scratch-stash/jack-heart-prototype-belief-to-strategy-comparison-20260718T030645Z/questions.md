# Confirmed interpretation and assumptions

- "Authority source digest" means managym's existing canonical
  `Env.state_digest()`. It covers canonical mutable match facts (including RNG,
  events, object identity, content digest, and allocation watermark), the
  current action space, pending choice, and skip-trivial authority surface.
- `Env.state_digest()` is the Study adapter's source-return witness, not a
  replacement for the search BranchDriver authority witness. The latter also
  includes the private structured-offer decision epoch for search admission.
- The return type stays inside `etude.study_branch`; the canonical replay row
  and persisted replay schema remain unchanged.
- Root drift is checked at both fork and return. A return-time mismatch consumes
  the branch and yields no recorded return payload.
- The approved second-serial review fix is intentionally limited to the exact
  digest receipt and focused retained-source/sibling-isolation proof. Broader
  stale/pack/nested stress and interactive performance evidence are not added
  in this PR.

# GAM-6: Prototype belief-to-strategy comparison

## Blockers (reported once, proceeding inline)

- 2026-07-17: `lf task acknowledge GAM-6 --directive 1` failed — the Loopflow
  registry on this machine was stale (`lf 0.11.3` vs a newer divergent local
  build; `lf doctor` reported migration `0.11.029_ci_incident_repaired_head`
  unknown) and the ambient `LF_WAVE_ID` pointed at a wave no longer in the
  registry. Resolved 2026-07-17: the compatible release binary
  (`LF_HOME=/Users/jack/.lf LF_BIN=/Users/jack/src/loopflow/target/release/lf
  /Users/jack/src/loopflow/target/release/lf`, `lf 0.11.3`, migrations
  applied/known both `0.11.029_ci_incident_repaired_head`) with
  `LF_WAVE_ID=game` recorded `GAM-6 incorporated directive v3`
  (`current_directive_version` 3, `incorporated_directive_version` advanced).
  No outstanding registry blocker.

## Assumptions (executive decisions, correctable in review)

- The two belief scenarios are modeled as two `StudyLandmark` rows at the same
  `erd1` decision inside one `StudyArtifact`, each with real flat-MC evidence
  under a disjoint seed family. The existing `StudyArtifact` contract permits
  this (no duplicate-decision-id check). A thin `scenarios` metadata array
  carries the prototype's presentation layer (labels, descriptions, inferred
  ranges) without touching the Rust-owned schema.
- The "inferred starting range" is a concise, honest, pinned text
  characterization per scenario — not an interactive range editor (the
  directive forbids generic range tooling).
- The live play surface loads the fixture's pinned completed-match decision as
  the demonstration advice surface beside the ActionPanel. The live frame
  during play has no `erd1` address yet (the canonical replay finalizes at
  game close), so live-address advice is a future GAM-4 integration point via
  the `onRequestAdvice` seam.
- The fork/Retry/return UI is out of scope (GAM-4's substrate, not yet
  publishable). The Study-mode component is advisory + comparison only.
- The shared action vocabulary is the frame's legacy `offers` (the live UI and
  the existing fixture already use it); the structured offer surface at the
  forked decision exposes only `pass_priority` and is not the vocabulary here.
