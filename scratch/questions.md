# Assumptions made without an answer (MTG-344 one-token layout)

Each is reversible before the PR lands.

1. The earlier object-record rename (`agent_objects`, `managym.Object` and
   their deprecated aliases) is dropped entirely: the branch tree returned to
   origin/main 452d7b5357 before the layout work.
