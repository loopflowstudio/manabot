# Open questions (MTG-319)

- `live_tokens` and `bucketed_shapes` are independent and combinable; the pilot
  measures both `live` and `bucketed-live`.
- CPU: NJT SDPA runs on CPU through a per-observation math fallback (correct, slow).
  Documented rather than rejected; revisit if a CPU stage should refuse it.
- `historical_mean` is not rejected (drafted docs rejected it): live attention
  returns zeros at padding exactly as the padded path, so its mean is unchanged.
