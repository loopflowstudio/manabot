# Open choices (assumed, reversible)

- Native frontier encoding is exposed as a `LocalUpdateTeacher` keyword and a
  top-level `make_player` spec key, not through `LocalSearchConfig` or
  registrations, because it changes no search semantics or receipt bytes.
  Registered evaluation players keep the Python path. Revisit if evaluation
  should opt in.
- Inference threads stay a process-startup choice (bench script flag), not a
  player or search parameter, following the frozen study's rule.
