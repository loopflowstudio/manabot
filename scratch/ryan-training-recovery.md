# Ryan's Jace training slow steps — recovery investigation

Follow-up on 2026-11-12: the local source repair and its successful scalar,
vector and bounded flat-search checks are recorded in
[`ryan-slow-step/unbreak.md`](ryan-slow-step/unbreak.md). The release-only
investigation below remains the original evidence; PyPI is still unchanged.

2026-11-12, incident graph step 1 (`unbreak`). Recovery remains blocked:
the supplied scalar workflow completes, but neither PyPI 0.2.1 nor 0.2.2
restores normal step latency. No engine, training configuration, checkpoint,
deck, or running training process was changed by this step.

## Original observation and recovery target

Ryan's report says: “It is not an infinite loop: every slow step observed did
return.” It reports 2–120 second native steps, versus about 4 ms median for
64 games, on Python 3.13 / Apple M3 Max / PyPI manabot 0.2.1. The supplied
426-action scalar reproduction reports 48.48 seconds on its final action.
Vector training and flat-MC playouts are affected in the report; only the
supplied scalar reproduction was exercised here.

Working means advancing the same game and legal actions without the reported
multi-second stall. The reporter's >0.25 second slow-step threshold is the
diagnostic check here, not a new performance guarantee. A checkpoint load or
unit-test pass would not establish restored training.

Original files remain untouched in `/Users/jack/Downloads/manabot-slow-step/`.
Exact copies are in `scratch/ryan-slow-step/`:

- `reporter-README.md`: SHA256 `cdf28dc970d1f7757c7f299bc0aa78f14e1afac54e05738a892fcdd77882b260`
- `reporter_manabot_slow_step.py`: SHA256 `42c6bec5f6a94fbb189ef40a92c7f197489e75756d3e813eaff063bd996596fa`

Checkout started clean at `48e958fc1a4faffe8e4e155efe6f013b71ab33b8`.
This step added evidence under scratch only. Another process prepared a local
venv/native extension during the investigation; those files and its evidence
were left alone.
At handoff, a concurrent 18-line diff also appeared in
`managym/src/flow/mana.rs`, changing the usable-mana bound to share a tap
resource. This step did not author, alter, or validate that patch. Its presence
does not change the release-replay results below.

## Actual reproduction

Local host: Apple M4 Max, macOS 26.0.1 arm64, CPython 3.13.15, uv 0.9.22.
Both completed replays ran the original script unchanged from its Downloads
path, with a 180-second subprocess timeout. No artificial load was induced.
Other work was running on this host, and the 0.2.1 replay included a one-second
stack sample; these are stall reproductions, not controlled version benchmarks.

| Installed package | Step 242 | Step 243 | Step 426 / maximum | Outcome |
| --- | ---: | ---: | ---: | --- |
| manabot 0.2.1 (bundled engine) | 0.73 s | 0.35 s | 33.65 s | All 426 actions returned; slow-step check fails |
| manabot 0.2.2 + managym 0.2.2 | 0.63 s | 0.32 s | 31.22 s | All 426 actions returned; slow-step check fails |

Logs: `ryan-slow-step/pypi-0.2.1-replay.log` and
`ryan-slow-step/pypi-0.2.2-replay.log`. Both log installed engine paths under
uv's cached `site-packages`, not this source checkout. Use explicit isolation
when repeating, because `--no-project` alone can still select a local venv:

```sh
uv run --no-project --isolated --python 3.13 --with manabot==0.2.2 python -I -u - <<'PY'
import subprocess
import sys

subprocess.run(
    [sys.executable, "-I", "-u", "/Users/jack/Downloads/manabot-slow-step/manabot_slow_step.py"],
    timeout=180,
    check=True,
)
PY
```

Substitute 0.2.1 to check the older package. Python's subprocess timeout kills
and reaps the child. The original 0.2.1 package bundles its engine under
`manabot.managym` and has no separate `managym` distribution metadata; an initial
metadata-only probe failed for that reason (`pypi-0.2.1.log`), before any game.

An initial 0.2.2 probe selected the concurrently created local editable engine.
It was terminated and reaped once that import path was noticed; its log is
`local-environment-aborted.log`. It is not evidence about PyPI 0.2.2 or a
completed current-source replay. The isolated retry above is the valid result.

## Evidence for the next debug step

`ryan-slow-step/pypi-0.2.1-sample.txt` is a one-second macOS `sample` of this
step's own reproduction child during the final stall. The 81 sampled main-thread
stacks run through:

```text
Env.step -> Game.tick -> Game.tick_priority
  -> Game.legal_activation_x_values -> payable_x_values
    -> Game.waterbend_payment_possible -> PaymentModel.plan
```

Most sampled leaves are destruction/freeing of payment search states and their
obligations. This sample points into search/state cleanup, not just construction
of `payment_model`; it does not prove the underlying algorithmic cause.
`managym/src/flow/mana.rs` is identical between release tags v0.2.1 and v0.2.2.
HEAD has later mana-resource and spending-restriction changes, so do not assume
release line numbers or action equivalence apply to HEAD without checking.

A separate bounded PyPI 0.2.2 replay stopped after the first 425 original
actions (30-second timeout). `pre-step-426-decision.json` retains its decision
frame: schema 8, revision 426, actor 1, seven activate-ability offers and one
pass-priority offer. Original next action is index 3. Fingerprint:
`74de04e581d5d2f2ee9b75d2a095ff4a894501723146fcd01781ab995f2e8502`.
An attempt to obtain a richer `capture_decision(0)` failed with
`_managym.PyAgentError: no active decision to project`; the saved frame is valid,
but no full viewer capture was obtained. See `pre-step-426.log`. Do not confuse
this additional diagnostic API failure with the original successful replay.

## Recovery decision and handoff

The attempted recovery was an isolated package upgrade, tested through Ryan's
exact scalar path. It fails the >0.25-second check at the same three steps.
There is no verified safe rollback or configuration workaround from this step.
Changing decks, suppressing mana choices, or resetting games would change the
training workflow and is not an established recovery.

Next debug step: reproduce against a correctly built HEAD extension, verify
the fixed indices still identify the same decisions, and investigate the
activation-X waterbend payment search with the retained release evidence.
Any repair must preserve legal choices and payment semantics, pass targeted
debug Rust checks, and return through the full 426-action Python reproduction.
Then verify the vector/flat-search paths before claiming Ryan's training is
restored. Ryan's actual Jace process, checkpoints, and live telemetry were not
available in the supplied evidence, so no claim is made about its current state.

All reproduction and sampling processes launched by this step have exited or
were explicitly terminated and reaped. No server or watcher was started.
