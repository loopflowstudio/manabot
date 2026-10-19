# Laptop evaluation stopped at Jack Heart's request

On 2026-10-18 Jack said: "please kill all evals on this laptop". This supersedes
protocol017's local execution placement. Do not restart an evaluation on this
laptop. Preserve the frozen protocol, raw checkpoints, and all partial results.

The operator sent SIGTERM to the sole protocol017 evaluation controller (PID
77023) at approximately 16:34 UTC. Its existing stop path stopped all four
process groups and admitted no further commands. The owning tool exited 1.
Fresh process inspection found neither the controller nor evaluation workers
remaining. Unrelated repository test processes were not experiment evaluations.

The controller's retained error is `Panel exited 124; partial results retained
without retry`. This was an explicit user cancellation, not a scientific failure,
deadline expiry, or model outcome. Its elapsed time was 1633.058 seconds. The
retained evaluation manifest was written. There are 44 complete deals / 352
games with passed replays; the planned 18,304-game panel is incomplete. Do not
choose a winning model from this interrupted panel.

GPU training had already completed: all 22 trajectories and 88 raw checkpoints
passed the retained-artifact and numerical checks, and both rentals were gone.
Those results are unchanged. Any remaining evaluation requires an off-laptop
execution plan that preserves the intended comparisons and accounts for this
cancelled attempt; no replacement has been launched.

Jack also asked to investigate his $53.64 monthly S3 charge and reduce its cost.
That is a separate infrastructure-cost outcome, not permission to delete
research evidence or alter the frozen training campaign.
