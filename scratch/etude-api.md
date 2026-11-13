# Historical Etude prototype

The full pre-EXP-56 gap analysis and stub design are retained at
`eb040c04f090ebfd331cdb1e9c4a7c251a7483e5:scratch/etude-api.md`.
It compared Etude main `be7c71b`; it does not describe the delivered cutover.
Current API observations are in cutover.md and delivery gaps in plan.md.

The prototype smoke used four streams and two updates, taking 79 seconds on
CPU. Its seed-197 PPO and damped-policy raw/EMA parity passed before the backend
merge. The post-merge proof in implementation.md supersedes that evidence.
The prototype wrote mutable final.pt and used a local Etude stub; both are gone.
