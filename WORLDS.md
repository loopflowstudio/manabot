# Worlds

An observation/action-shape change is a **world version**. Checkpoints,
shards, and reports are only comparable within a world; cross-world numbers
never share a table untagged. New worlds are frozen deliberately (batch the
shape changes), and on each freeze the headline baselines are re-run before
new claims are made. (Cost of ignoring this, measured: every exp-07 artifact
was dead on arrival in exp-10.)

| world | frozen at | shape (player / card / permanent / action-types) | live artifacts | headline baselines |
|---|---|---|---|---|
| **w0** | pre-2026-07-09 (first-light era) | 26 / 29 / 5 / 7 | none (all superseded) | exp-04 terminal-only 60–75% |
| **w1** | rules stage 1–2 (`2d124c9`..`50e0a1f`) | 27 / 37 / 7 / 14 | none (exp-07 artifacts dead per exp-10) | exp-07 student_r0 87%, ladder ≈N=7 |
| **w2** | rules stage 3–4 + conformance (`cb80331`..`a9f1f91`, max_actions 32 @ `55a0b4b`) | 28 / 38 / 24 / 14 | exp-10 V + BC student; exp-11 arms (incl. ported student_r0, validated 86.5%) | exp-06 PPO 60–77%; exp-10/11 (pending merge) |

**Current world: w2.** Tag = the world column; when in doubt, the dims tuple
is the tag. Porting across worlds (`port_legacy_state_dict`, exp-11) is legal
for *components* (opponents) after behavioral validation; *measurements* are
regenerated, never ported (exp-10's precedent).

Update this table in the same PR as any shape change.
