# Clean-machine play proof

Manabot has one clean-machine-certified play command:

```bash
./scripts/play
```

From a fresh checkout on the reference profile, that command prepares the
locked, minimal play-runtime Python environment, builds the CPython 3.12 `managym` extension,
installs the locked frontend dependencies, starts the local FastAPI and Vite
processes, validates the installed curated pack, and prints a
`MANABOT_PLAY_READY` JSON record plus the play URL. Ctrl-C stops both child
processes.

The existing `uv run scripts/play.py` path remains useful in an already
prepared development environment. It is not the clean-machine contract because
it cannot diagnose a missing `uv` before Python starts.

## Reference profile

The repeatable proof runs on a fresh GitHub-hosted `ubuntu-24.04` x86_64
checkout with:

- `uv` 0.9.22 and provisioned CPython 3.12;
- Node 22 and its matching npm;
- the stable Rust toolchain, Cargo, and the runner's native build tools; and
- the Chromium revision installed for Playwright 1.61.1.

The JSON receipt records the actual OS release, CPU, memory, tool versions, and
Chromium version observed on each run. The hosted runner image may evolve; the
recorded facts are evidence and are not silently replaced by the label above.

Before timing begins, the checkout must not contain `.venv`, `.cache`,
`frontend/node_modules`, generated Svelte/Vite/build/test output,
`managym/target`, a `managym/_managym*.so` extension, GUI trace JSON, or a
prior proof browser profile. The proof job restores no uv, npm, Cargo, target,
or project-install cache. Installing the host tools, browser, and checkout is
provisioning and is outside the measured command.

## Sixty-second boundary

The wall clock starts immediately before `./scripts/play` is spawned. It ends
only when a fresh Chromium context shows all of the following:

- a connected WebSocket;
- the default Search 64 opponent;
- `UR Lessons vs GW Allies`;
- at least one enabled authoritative legal action; and
- pack treatments for every visible card, with no fallback treatment.

The measured time includes uv's locked play-runtime sync, the release native build,
`npm ci`, both service startups, browser navigation, and default match
creation. It must be strictly less than 60,000 ms. The launcher-internal time
is diagnostic; the externally measured receipt is authoritative because it
also includes uv startup and dependency work.

Run the complete proof only from a checkout that satisfies the empty-artifact
predicate:

```bash
./scripts/verify-clean-machine
```

The verifier exercises `./scripts/play` once, runs only the focused browser
scenario against those processes, writes `.cache/clean-machine/proof.json`,
and tears the process group
down. CI uploads the receipt and launcher log even if the proof fails.

## Offline reload boundary

Package registry access is allowed while the first command installs locked
dependencies. After the launcher is ready, Playwright denies every non-loopback
`http`, `https`, `ws`, and `wss` request before navigating to the table.
Loopback HTTP and WebSocket remain available because the authority is local.

The proof starts the exact pinned matchup, takes a deterministic legal action,
and records the session plus a canonical visible board/prompt/action signature.
It reloads with public networking still denied and requires the same session,
pack reference, matchup, visible state, and pack-only treatments. A second
legal action must advance the authority after reload. This proves offline
runtime assets and recovery; it does not claim that dependency installation or
the local authority can run without networking or a local process,
respectively.

The pack authority remains
`frontend/src/lib/packs/tla-ur-lessons-vs-gw-allies/v1/manifest.json`. Both the
launcher and browser receipt derive `{id, version, manifest_sha256}` from its
installed bytes. There is no duplicated deck list or expected hash in the
proof.

## Deterministic diagnostics

Controlled failures print one `MANABOT_PLAY_ERROR` JSON record and exit
nonzero. The stable codes are:

| Code | Meaning |
| --- | --- |
| `prerequisite.uv` | `uv` is absent |
| `prerequisite.node` / `prerequisite.npm` | the frontend runtime is absent or unsupported |
| `prerequisite.rust` / `prerequisite.cargo` | the native build toolchain is absent |
| `lock.python` / `lock.cargo` / `lock.frontend` | a required lock is absent or rejected |
| `python.version` | the launcher is not using CPython 3.12 |
| `pack.missing` / `pack.invalid` / `pack.notice` | the installed curated pack cannot pass preflight |
| `native.build` / `native.import` | the CPython 3.12 extension cannot be built or imported |
| `frontend.install` | `npm ci` failed or could not be recorded |
| `port.in_use` | a requested loopback port is unavailable |
| `backend.start` / `frontend.start` | a child exited before or after readiness |
| `ready.timeout` | the bounded local readiness probe expired |
| `launcher.internal` | an unexpected startup boundary failed before readiness |

A damaged curated manifest fails closed before either service starts. The
W2-185 deterministic fallback remains valid for an out-of-pack identity in a
legacy/custom game; it is not used to conceal a missing identity or treatment
inside the exact v1 pack. The launcher never fetches card data or art.

## Supported and excluded behavior

The certified slice is one local session, one Chromium context, UR Lessons as
hero, GW Allies as villain, Search 64, and the installed v1 pack. The launcher
also retains `--port`, `--frontend-port`, and `--no-frontend` for development.

This proof adds no deck builder, arbitrary-card fetcher, service worker, PWA
cache, authority adapter, browser authority, or WASM. It is not a production
installer, deployment artifact, Windows certification, or promise that a bare
operating system reaches play without provisioned host tools.

The minimal clean environment certifies Search 64, random, and passive local
play. A policy-checkpoint opponent is a training-stack feature and still
requires the repository's full development installation; it is not part of
this clean-machine profile.
