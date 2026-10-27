# OpenRouter evaluation publication

Jack Heart requested the OpenRouter observation adapter, cost/caching analysis,
then a pilot and 100 high-effort Haiku games against MTG-183. The finished result
is Haiku 51–49, $37.0833537, 66,275 exactly replayed Commands, five repairs and no
fallbacks/failures. Durable protocol, caveats, matrix and cache instructions:
`experiments/haiku-mtg183.md`. Adapter usage: `docs/openrouter.md`.

Jack Heart now authorized cleanup, committing, experiment-cache publication,
and a landed PR. Applying compress and pr-land; no Task or new checkout.
Full prior local notes preserved at
`.runs/openrouter-cost/publication-notes-original.md` before reduction.

Frozen evidence is untouched under `.runs/haiku-pilot`, `.runs/haiku-ten` and
`.runs/haiku-hundred`. The lossless archive and aggregate copies are staged under
`experiments/data/haiku-mtg183-20261027/` for `manabot.data publish`. Never track
these bytes in git; commit only data.manifest.json and the report. Archive:
369,119 files, 1,040,916,791 compressed bytes, SHA-256
`daa2f074f5c03723a12a1e032f62287ea3447fea36504c8923577b4f5416c656`.
The package scan found no OpenRouter keys or long Bearer-token patterns.

Cleanup separates hosted billing from the game loop and records checkpoint/setup
failures in final summaries; it does not reinterpret or rerun the paid experiment.
Current evidence: 90 focused tests passed; 13 artifact/cache tests passed in an
isolated environment; optional hosted import passed without Torch/Transformers
imports; exact main-CI selection collected 688 tests; Ruff and diff checks clean.
The documentation's tests/remote/ path is stale/absent in this checkout; current
artifact/cache tests were used for isolated dependency coverage.

Published all six artifacts and restored all six into a fresh local directory
from the shared cache, matching every digest/size with zero remote downloads.

Pending: land via
`lf land --wait-and-fix` with concrete PR copy; report verified merge and artifact
location. R2 credentials come from Doppler manabot/prd. OpenRouter key location
is loopflow/dev_personal; no more paid inference is needed for publication.
