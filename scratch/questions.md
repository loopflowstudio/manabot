# Card evaluation design — 2026-10-23

- Jack Heart selected card ranking by GIH as the first result, with future
  17Lands-style API expansion and fidelity in the data-processing path.
- Implementation design: `scratch/17lands-for-bots.md`; investigation:
  `scratch/17lands-for-bots-evidence.md`. Jack Heart subsequently requested the
  pursuit after review; the dated implementation defaults below settle engineering choices.
- Engineering defaults: CLI/SQL first, one subject model
  and explicit evaluation scope, existing authored deck layouts, true draws
  excluded and reported separately, no implicit minimum-sample filter.
- Arena compatibility initially means a versioned, offline-tested log subset
  parsed by the pinned public 17Lands client. Full downstream metric parity
  remains unproved without reference fixtures; copying parser hand sightings
  directly into GIH would be incorrect.
- Native cards need a verified mapping to Arena printing IDs for export.
  Missing mappings cannot be invented. Rankings use content-bound native IDs.
- No Wave or Task was named; standalone design needs no placement lookup.

Prior MTG-176 recovery assumptions, sync provenance and remote-proof limits
remain at `003644bda6d3ed2e1e6754186960b135ff815e76:scratch/questions.md`.
They do not select current work or authorize paid execution.

## GIH implementation defaults — 2026-10-23

- Jack Heart's launch request supersedes the earlier design-only note. The
  keystone keeps CLI/SQL, copy-weighted draws, and the existing eight-leg authored
  matrices; no later metrics, API, UI, paid execution or external submissions.
- Installed `etude-sdk[eval]==0.2.0` supports the inspected evaluation APIs.
  Its delivery only includes metrics/checkpoints/tables/matches, not arbitrary
  artifacts. Evidence therefore lives under the evaluation's `evidence/`;
  explicit evaluation publication copies and verifies these immutable bytes via
  Etude Storage before delivering Etude's indexes. Plain `etude sync` alone is
  not a card-evidence publication command. This fills a concrete SDK gap without
  another match scheduler, database, or evidence store.
- Use the existing `evaluation` extra in commands. Confidence intervals resample
  whole deal seeds across all selected pairings. Native capture is private,
  setup-enabled and absent from clones. No rule/world version change is intended.
- The pinned public client reports mulligan counts of -1 when no mulligan prompt
  exists. Preserve that unavailable value; do not invent a mulligan decision.
  Exported logs document this bounded subset and cannot certify server metrics.
- Scheduled-admission default (2026-10-23): freeze the native content digest
  alongside world, engine source and content-manifest identities in the existing
  unreleased Etude environment. Missing identity fails admission; never infer it
  from the reader's runtime or rewrite historical evidence. Deck/seat validation
  uses the existing authored matrix, including complete deck and sideboard setup.
