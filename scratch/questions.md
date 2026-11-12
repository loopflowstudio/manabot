# Open choices (reversible, settled for now)

- Switch: environment variable `MANABOT_BUCKETED_SHAPES=1`, read when `ActorCritic`
  is constructed. Chosen because Etude/L4 jobs can A/B it without a ModelConfig field.
  Any value other than unset/""/0/1 raises.
- Buckets: powers of two, minimum 8, capped at the declared capacity.
- Catalog GRU: full catalog in learning under the switch (counts in plan.md).
- Not converted to fixed shape (option 3 did some): fact program GRU (option 3's
  per-slot loop would be bucket-length serial GRU steps), shared-effect CSR
  routing (host-built), legal-action selection. Candidates for a follow-up if the
  L4 profile shows them.
- Coordination: MTG-299 (validation to admission) and MTG-300 (catalog cache) edit
  the same forward. Conflicts expected in `actor_critic.py` catalog selection and
  in encoder shape checks; the count keywords are the merge point.
