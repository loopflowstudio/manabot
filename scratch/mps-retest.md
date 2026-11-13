# MPS retest implementation

Build the capture → correctness → per-shape timing → whole-update replay →
summary/chart path described in `experiments/mps_retest.md`, including all
steers quoted in questions.md. No automatic backend-adoption threshold.

Delete — do not maintain (implemented):
- Pilot-local capture hooks moved to `manabot/backends/capture.py`; retain
  the original file format and CLI. Use the current learner optimizer API.
- MLX's model-static ownership sign removed; ownership follows each prepared
  batch's trimmed axes. Checkpoint buffers and padded results survive.
- Replace half-dtype public logits with float32 logits at the head boundary;
  float16 cannot hold the existing -1e8 illegal-action sentinel. Gradients flow
  through the cast; regime precision refusals and defaults remain unchanged.

Remaining:
1. Resolve the capped AWS M2 launch/release path. Do not allocate uncapped AWS
   infrastructure or substitute noisy laptop measurements.
2. Run the committed harness on the rented M2, then only genuine overflow cells
   on the M4 Max under load 8. If laptop timing is needed, tell the operator the
   exact block and observed memory need, then pause until it is quiet.
3. Complete the expanded matrix and same-hour triple replay groups; inspect
   actual half-precision behavior and fix real runtime failures, not label
   implementation defects as unsupported dtypes.
4. Publish retained evidence with manabot.data, embed full_context.png, write
   the answer and reasoned recommendation, and update the backend README.
5. Gate owns the affected suites and experiment acceptance checks. Publication
   and PR handoff belong to later Graph steps; put full_context.png first in
   the eventual PR body.

Checks: locked Python 3.13 sync succeeded; focused MLX trimming test 1 passed;
float16 output-boundary test 1 passed; experiment discovery/backend boundaries
3 passed; controlled two-row CPU/MLX runner and incomplete-report fixture
passed without collecting timing. Full MPS/remote acceptance has not run.
