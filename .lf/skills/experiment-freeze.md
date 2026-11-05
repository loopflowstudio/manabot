---
requires: reviewed protocol and admitted integrated software
produces: frozen source/protocol identity and reconciled execution bounds
---
Bind the reviewed experiment to its final integrated executable source.

1. Verify any software delivery is integrated before freezing. Bind exact source,
   world/input, manifest, seeds, model/regime, checkpoint roles, expected schedule,
   metrics, analysis and inclusive resource bounds. Use existing Etude declarations
   and admission; inspect discovery without executing games or training.
   For a growing-pool revision, freeze one `ProtocolBinding` in a thin importable
   experiment entry point. Its module stem names the Etude experiment; its path
   and digest select the protocol. Verify discovery, TrainingJob, canonical
   worker callback and offline analysis all select that binding. Include the
   manifest, entry point and shared runtime in frozen source; a path-only check
   cannot establish this contract. Reject crossed protocol/experiment identities.
2. Check held-out isolation and stable anchors. Changes after freeze require a
   prospective protocol revision and newly admitted controls; never relabel frozen
   data or checkpoints to cross incompatible worlds.
3. Retain the explicit execution authority and its bounds, including any existing
   paid-run authorization, separately from proposals. Missing approval or required
   scientific judgment remains unresolved at execution review.
4. Keep a single receipt in the owning protocol/design with digest and admission
   result. MTG-245's reference protocol, splits and frozen evidence remain intact;
   later integrated batches create their own prospective bindings.
