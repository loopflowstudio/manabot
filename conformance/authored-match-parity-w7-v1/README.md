# Live, headless and persisted replay parity — w7

The w7 authority tape executes identically across all three surfaces:
156 Commands, 157 checkpoints per surface and 942 viewer projection checks.
The proof also rejects a retained object candidate and a stale revision
without changing the root witness or event cursor. Both player replay
projections preserve private hands; spectator projection is rejected.

```bash
uv run --extra dev python -m academy.authored_match_parity verify
```

Verification compares every field, including the current source manifest.
The older `authored-match-parity-v1` receipt remains unchanged. Historical
RUL-9 and public-commitment contracts retain their original inputs; this proof
does not recertify their measurements or close their current-runtime failures.
