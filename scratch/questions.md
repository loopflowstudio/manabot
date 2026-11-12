# MTG-327 questions

- Two reproduction revisions are named: the config error says 439379869 (last
  main revision with pooled); the report says 7ac8c9b6f (its pilot revision).
  Both load a pooled configuration; left as is.
- `_embed_actions`/`_focused` (split out for pooled) are folded back into
  `_gather_informed_actions`; `EmbeddedObservation.relations` removed.
