# CI demo decision — 2026-11-12

Historical decision for #506, superseded by the failed merged push recorded in
`ci-post-merge-verification.md`. The current repair and outstanding validation
are in `ci-under-8-minutes.md`; this note does not validate the fix-forward.

No human demo happened: this step ran headlessly with nobody present. There
are no reactions or approval from Jack Heart to record.

Decision: advance on the measured outcome. The preceding pass sought a green
PR run under 480 seconds per ordinary Ubuntu job. GitHub now reports PR #506
run [43337105047](https://github.com/loopflowstudio/manabot/actions/runs/43337105047)
as a successful `pull_request` run at the then-current head
`1fc4e1b244f856e1a4192cc461c0edeabf0a7115`. Every ordinary job passed;
recovery was slowest at 349 seconds including setup, leaving 131 seconds.
The exempt Engine Budget passed in 613 seconds. This closes the earlier
PR-validation evidence gap. Main's recorded slowest Integration job was
1443 seconds. Prior QA retained coverage and verified the worker watchdog.

No concrete implementation gap warrants returning to implement. Timing varies:
the preceding full dispatch reached 448 seconds, so this run is evidence of
meeting the target, not a guarantee for every runner. No demo processes were
started. This decision advances the Graph; it does not merge or complete the Task.
