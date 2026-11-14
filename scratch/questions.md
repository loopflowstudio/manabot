# MTG-329 open work

Historical details: 382c653ec8:scratch/questions.md; earlier exact remarks:
bd14801a72:scratch/questions.md.

Reversible choice: start= takes a completed CheckpointRecord, not a Job.
Finish belief_policy, then declare belief_sampler from its final checkpoint.
Automatic same-seed scheduling still needs product review and Etude support.
The real local probe failed `input not given: 'start'` after policy success;
0.4.0 local._prepare still omits job inputs/environment. No paid fallback tried.
Games in flight, random streams and in-run source repair are not recovered.

Context exception: the Task steer says "Keep the scratch notes
(design-train-api.md, ontology.md, positioning.md, exp-56-compat.md, sketch.html,
explore.html, etude-readme-draft.md); do not delete or shorten them."
Preserve those notes even if they exceed the compress target; do not raise limits.

Compression budget conflict: protected notes remain untouched. The final
scratch overage is recorded by `lf context`; the byte limit is met. Further
cuts would shorten protected context or discard live contrary findings.
Limits are unchanged.
