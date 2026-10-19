# Correct Mini launch scheduling and worker event records

The first 32-game control was launched at 17:06 UTC with launchd ProcessType
Background. Its workers ran at priority 4. By 17:12, retained traces showed two
completed greedy games and six game-watchdog failures across the four commands.
The operator stopped the controller rather than continue a control that could
not pass. It exited with retained failure after 325.735 seconds; all descendants
were gone. Original admission, traces and logs remain in mini-calibration.
This is an operational control failure, not a model-size result.

The installed launchd.plist manual says Background throttles resource use for
work not directly requested by the user. That was the wrong classification for
this requested compute. A separately named replacement control uses Standard,
with the same 32 games, source/runtime/checkpoints, 900-second command limits,
1,200-second work limit and 240-second retention reserve. No game watchdog is
relaxed. Whether normal scheduling resolves the failures remains to be measured.
No protocol017 learning-results game has run on the Mini.

The control also exposed interleaved supervisor JSON events. PR #311,
https://github.com/loopflowstudio/manabot/pull/311, merged at 17:38:02 UTC with CI
passing. Commit cbf2ebde4fd58dbccdecf4192d610fc98b1c7bef adds a shared output lock
around each printed event and newline. A four-thread regression reproduced the
original JSONDecodeError and all four concurrency/cancellation tests passed.
This changes supervisor logging only. The Mini retains the original scientific
source and an explicit, hash-bound patch to experiments/runners/gpu_job.py.
Its game evaluation runtime, policies, binary and inputs remain unchanged.

The original control source is preserved separately in mini-transfer before
updating the controller for the Standard attempt. Both attempts and the
operational patch must accompany the final research handoff. Do not silently
replace the original control, retry again on failure, or infer scientific
equivalence before the complete decision/replay comparison passes.
