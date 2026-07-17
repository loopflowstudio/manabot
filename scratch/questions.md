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
