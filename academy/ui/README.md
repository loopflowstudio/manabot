# Academy UI

Academy's Svelte browser client lives here; the Python session and protocol
owners live in the parent `academy` package. Start the complete local app from
the repository root with `./scripts/play`, or `./scripts/play --demo precision`.

From the repository root:

```sh
npm --prefix academy/ui run check
npm --prefix academy/ui test
npm --prefix academy/ui run build
npm --prefix academy/ui run test:e2e
npm --prefix academy/ui run test:e2e:release
npm --prefix academy/ui run fetch:card-art
```

Scryfall images are cached locally and bundled when present; the app uses
procedural treatments when absent. Artist and source credits are in the footer.
Generated dependencies, images, builds and test output are ignored by git.
Python wheels include immutable asset packs, without the Node application or
its dependencies. Pack identities and historical protocol evidence keep their
original bytes when moved.
