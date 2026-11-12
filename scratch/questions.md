# MTG-327 questions

- PR #495 (MLX backend) is still open; implemented on main as is. When it
  lands, drop "pooled" from `manabot/policy/checkpoint.py`'s refused list.
- The removal message names commit 439379869 (last main revision with pooled)
  as the reproduction point; the report also pins its own revision.
- `_embed_actions`/`_focused` (split out for pooled) are folded back into
  `_gather_informed_actions`; `EmbeddedObservation.relations` removed.
