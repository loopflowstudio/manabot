# MTG-290 decisions

Reversible API spelling assumption: `managym.observation_layout(version,
policy_history_version=0)` returns native dictionary metadata;
`managym.recommended_policy_input_version()` returns 9. Implemented with
column names, indices, kinds, encodings, ranges, axes, fixed extents and tagged
row descriptions. Unknown row capacities are `None`, not implicit defaults.

## Owner proposal (non-blocking)

Should fresh `ObservationConfig()` and `Env.encode_observation(obs)` default to
v9, or remain v6 with an explicit v9 recommendation for new consumers?

The task brief says: "Changing the fresh default itself is the owner's call;
propose it with the consequences if it is the right answer, do not change it
silently."

Changing to v9 would let implicit fresh consumers represent current layer and
execution programs and name choices, but adds shared-effect/name arrays,
increases allocation, changes fresh model architecture and serialized config
identities, and requires consumers that assume v6 keys/shapes to adapt. It does
not eliminate finite-capacity failures. Saved versions, omitted-version v1
loading and frozen evidence must remain unchanged. No owner decision to change
the default was found; implementation and docs retain v6. The PR carries the
proposal; the Task's done-when does not depend on the answer.

The task title says "accepts every legal board". Even v9 has finite capacities,
so that literal guarantee is not implemented or verified. The brief's concrete
acceptance asks for named layouts, version guidance and improved errors. Demo
should clarify whether that documented contract satisfies the title; unlimited
board acceptance would require further capacity design, not just a default bump.
