# Questions

- `content/semantic/current/learning_schema.json` still says
  `managym.learning_token_value` in its `token_values` description. Left unchanged:
  the schema JSON is hashed into semantic-pack bindings (`_hash_json(raw)` in
  `manabot/semantic/learning.py`), so editing the text would move the binding hash.
  MTG-294 (alias removal in 0.3.0) should decide whether to reword it at a freeze.
