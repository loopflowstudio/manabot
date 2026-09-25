# Learn visual references v4

Generated and reproduced on 2026-09-25 using Ubuntu 24.04 x86-64, Node 22.16.0,
Playwright 1.61.1 and Chromium 149.0.7827.55. The local Linux container ran from
`mcr.microsoft.com/playwright:v1.61.1-noble` at digest
`sha256:5b8f294aff9041b7191c34a4bab3ac270157a28774d4b0660e9743297b697e48`,
with the viewport, fonts, locale and rendering settings in the active
`frontend/e2e/release-prompt-matrix.json`.

All three release browser tests passed while generating the references, then
passed again with strict screenshot comparison. The matrix contract also
passed. The 17 images cover nine prompt families, three board states, three
reconnect states and two terminal results. Learn's discard selector displays
actual card previews. Both seeded games now end in a hero loss; the prior GW
win remains in the historical v3 corpus alongside its original matrix.

On the pinned Linux profile, after installing the locked runtime and building
the native extension:

```bash
npm --prefix frontend run test:e2e:release -- --update-snapshots
uv run --no-sync pytest tests/etude/test_release_prompt_matrix.py -q
npm --prefix frontend run test:e2e:release
```

Version new accepted visual or rules changes instead of replacing historical
corpora. A run with ignored screenshots does not certify these references.
