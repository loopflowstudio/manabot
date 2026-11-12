# Assumptions (reversible)

- Unadmitted forward raises `TypeError` rather than admitting implicitly. Implicit admission would keep
  validation host reads inside forward for every direct caller, which the Task's outcome rules out.
- Admitted observations are bound to the admitting `ModelConfig` + observation config, not to the
  model instance, so a bootstrap/target agent with the same config can reuse a learner batch.
- Learning admits only the rows an update trains (plus bootstrap rows), not rows the advantage filter
  drops: those never entered a forward before either, and admitting them would add CPU work.
- Test doubles that are not ActorCritic (`TinyPolicy`, the likelihood `_FakeAgent`) stub admission in
  their test modules; they read no policy input.
- `admit_observation` accepts tensors on any device but is documented and used on CPU; device models
  use `admit_on_device`.
