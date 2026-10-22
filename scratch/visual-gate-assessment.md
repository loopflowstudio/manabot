User direction, 2026-10-21: aggressively path-filter CI, merge main, and delete the slow visual tests. This supersedes the recommendation to retain the full-match matrix on demand.

Deleted the full-match screenshot spec, its active matrix, exclusive Python contract and unshipped v8 images; removed the update workflow input and visual CI job. Kept historical v1–v7 evidence and focused browser interaction/accessibility tests. Job selection now isolates frontend, backend, training, semantic, website and launch paths, including main pushes; unknown/shared infrastructure paths run all jobs. Tests Result remains the required aggregate and rejects unexpected skips.

Sync recovered from rewritten history by first merging the published branch (same file tree), then main 43d80703. Adapted native Decision::Payment explanation, moved the new immutable Elemental v2 manifest into academy/ui, and captured new explicit w11 demo/navigation prefixes through actual Commands. Prior world evidence remains unchanged.

No hosted deploy or training launch. Ship/merge/completion remains authorized. Auto-merge disabled while incorporating this direction; re-arm only after checks and PR description reflect final scope.
