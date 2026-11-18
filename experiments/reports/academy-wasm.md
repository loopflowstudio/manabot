# Academy: engine in the browser (WASM) or the Python server

Exploratory, 2026-11-18. Should Academy run the managym engine in the browser
as WebAssembly, or keep the Python server it uses today? Nothing was migrated;
this report compares the two and recommends one arrangement.

## Answer

**Keep the Python server as the authority for everything that acts on a game:
playing, watching, trying another line, and turning a game file into a
recording. To look at a finished game without a server, export the recording's
positions as static files the existing client reads. Do not put the engine in
the browser now.**

The engine compiles to WASM unchanged and plays the same games as the native
build, about twice as slowly and still under 0.1 ms a step. That is not what
decides it. The engine is a small part of what answers a Command: Python
around it walks records, projects each viewer's frame, derives presentation
events and binds Commands, and none of that has a browser home. A WASM engine
alone can produce a state digest; it cannot produce one frame the client can
draw. WASM becomes the right tool only if that session layer moves from Python
into the `managym` crate, which is a larger decision this report leaves open
([Open decision](#open-decision)).

## Rerun it

```bash
rustup target add wasm32-wasip1
uv run python experiments/academy_wasm.py
```

The script builds `managym/examples/bench_steps.rs` (the 512 curated games the
Engine Budget already runs) natively and for `wasm32-wasip1`, runs the WASM
build under Node's WASI, records one fixed-seed game and reads it back through
Academy's own recording code, and prints one JSON line. Wrapped here:

```json
{"baseline_digest": 3471185654390571689,
 "native": {"digest": 3471185654390571689, "steps": 124687, "steps_per_s": 14200},
 "wasm": {"digest": 3471185654390571689, "steps": 124687, "steps_per_s": 11300,
          "bytes": {"raw": 5250000, "gzip": 1300000, "brotli": 860000}},
 "recording": {"positions": …, "seconds": …,
               "one_seat": {"raw": …, "gzip": …},
               "both_seats": {"raw": …, "gzip": …}}}
```

| Field | Meaning |
|---|---|
| `baseline_digest` | `bench_steps.digest` in `managym/perf/baseline.json`: the digest over every decision of the 512 games |
| `native`, `wasm` | The same benchmark from each build. Both digests must equal the baseline |
| `wasm.bytes` | The `.wasm` file holding the engine and every card set, without `wasm-opt` |
| `recording.positions` | Decisions in the recorded game, each one a position the client can open |
| `recording.one_seat`, `both_seats` | Bytes of the recording index plus every position, as one seat saw the game and with both hands revealed |
| `recording.seconds` | Python time to produce every position for one viewer |

It exits 1, naming both digests and step counts, if either build disagrees
with the baseline. It needs rustup's toolchain and Node 24 or newer, and says
which is missing:

```text
academy_wasm: rustup has no wasm32-wasip1 target; run `rustup target add wasm32-wasip1`
academy_wasm: node 24 or newer is required for node:wasi; found none on PATH
```

A `cargo` that rustup does not manage (Homebrew's) has no WASM target and is
never used. The first run takes about three minutes per build. The benchmark
is the fixed workload CI budgets; the script adds no load and is not a CI job.

## What a person does, and what each needs

A person never meets the engine. They do five things in Academy, and the
answer differs for each.

| What the person does | Needs the engine | Needs an opponent policy | Hidden cards must stay away from them | Recommended home |
|---|---|---|---|---|
| Play a game against a bot | yes | yes | yes when hosted | server |
| Watch a game a loop is producing | yes, where the loop runs | yes | per viewer | server |
| Step through a recorded game | no: a stored recording is positions, and reading one never executes a game | no | per viewer | static export, or server |
| Try another line from a recorded position | yes | sometimes | same as the recording | server |
| Open a game file (seed, setups, actions) for the first time | yes, once, to turn it into positions | built-in seats only | no: the file holds both hands | server |

Stepping through a recording is already two reads, the recording index and
one position per decision (`/api/traces/{id}/recording` and
`/api/traces/{id}/positions/{key}`). Their answers are fixed once the game is
finished, so they can be written out as files.

## The three arrangements

| Concern | Python server (today) | Engine in the browser | Exported positions, no engine |
|---|---|---|---|
| Latency | 16–30 ms median from Command to answer on one machine; hosted adds a round trip | No round trip. The rest of the 36–62 ms is Python that would have to be ported first | Stepping is a file read |
| Hosting | One stateful machine, since live games are held in process memory | Static files for whatever runs in the page | Static files |
| Offline | A local install | A cached page | A cached page or a folder |
| Policy inference | In process through `managym.bots`; Torch stays out of Academy | Random, heuristic and search could run in WASM. A trained policy needs a server call or exported weights | None: recorded policy evidence is in the positions |
| Recordings, replay and fork identity | The server writes recordings and vouches for them | Holds for deals, shuffles and rules. Built-in random and search seats need the fix below. A record written in the browser is unverified | Read-only; the recording's digests travel with it |
| Command authority, stale rejection | The server binds each Command to the frame that offered it | The same check in the same tab guards against bugs, not against the player | No Commands |
| Viewer privacy | Hidden cards never leave the server | A live game ships the seed, both hands and both libraries to the page | One seat's view, or both by choice |
| Download | 153 KB gzip of JavaScript | Plus 0.86 MB (brotli) to 1.30 MB (gzip) of engine | Plus the positions: `recording.one_seat.gzip` |
| Another build's games | `replay.rejected` names the build to use | The same refusal: the page is pinned to one world | Readable in any later client |
| Try another line | yes | yes | no |

### Why not a WASM replay viewer

It sounds like the safe middle: the holder of a game file already has both
hands, and re-execution needs no authority. But re-executing a game into
something viewable is the engine plus about 6,000 lines of Python walk and
projection (`managym/replay.py`, `recording.py`, `experience.py`,
`experience_projection.py` and `managym/academy/replay_index.py`,
`recording.py`, `trace.py`, `presentation.py`). Exported positions give the
same stepping with none of it, and can share one seat's view, which a file
carrying the seed cannot.

### Same games in both builds

The WASM build produced the baseline digest over all 512 games and 124,687
steps. It did so only once the benchmark's chooser drew its random index as a
`u64`: `gen_range` over `usize` draws a different stream on a 32-bit target
(120,134 steps and another digest before the change). The engine's own
built-in seats draw the same way (`Env::random_action_index`, search playouts,
the vector environment's random seats), so a recording whose seat was Random
or Search re-executes identically across builds only with those draws as
`u64`. On 64-bit that is the same stream, and native digests do not move.

### Where a Command's time goes

After the save fix, a Command costs 36–62 ms of server time: persistence
16–24, projection 7–15, the opponent 8–18, engine step plus recording 11–25.
The engine step itself is about 0.04 ms native and 0.085 ms in WASM. Moving
only the engine removes the network round trip and nothing else.

## Open decision

Move the walk, projection, presentation and Command binding into the `managym`
crate, with the Python server and a WASM build as two thin hosts? It is the
only route on which browser play or browser re-execution is cheap, and it
would also remove the 7–15 ms of projection per Command on the server. It is a
rewrite of roughly 6,000 tested lines. The measurement that would justify it:
projection and recording as a share of Command time on the hosted machine,
with a person waiting on it.

Two questions about a browser engine stay with it: whether hidden information
matters when the only possible peeker is the person playing a built-in bot on
their own page, and whether trained opponents should ever run from exported
weights.

## Proposed follow-ups

Not filed.

- Export a finished recording as static files the client opens, for one seat
  or with both revealed.
- Restore or retire hosted play. `play.manabot.gg` and `academy.manabot.gg`
  return Cloudflare 525: the `academy-play` app has had no machines since its
  2026-10-14 release.
- Decide the open question above.

## Cost

USD 0. One laptop.

## Limits

- One M4 Max laptop with other builds running. Native speed ranged from 3.0k
  to 14.2k steps/s across runs and WASM from 5.9k to 11.3k; only the ratio and
  the order of magnitude are claimed.
- WASM ran in Node 24 (V8). Chrome should match; Safari and Firefox were not
  measured.
- The size is a `wasm32-wasip1` benchmark binary. A browser package
  (`wasm32-unknown-unknown` with bindings) was not built; it fails today only
  in `getrandom 0.2`, which needs its `js` feature.
- Command timings are quoted from the Academy table rebuild's measurements on
  the same laptop, not re-measured here.
- The hosted round trip could not be measured while hosted play is down.
- Today's Python session running in the browser on Pyodide was not measured.
  It would reuse the Python and ship an interpreter of tens of MB, with the
  same privacy and authority losses as any browser engine.
- The export is sized, not built.
