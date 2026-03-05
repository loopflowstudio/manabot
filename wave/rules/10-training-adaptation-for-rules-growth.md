# 10: Training Adaptation for Rules Growth

## Finish line

Training remains stable as mechanics expand, with lightweight instrumentation
and staged mechanic rollout in configs (no heavy process overhead).

## Changes

- Track invalid-action rate, action-space truncation, episode length, and
  termination/truncation mix per milestone.
- Update observation/action encodings for new target/mechanic classes.
- Add lightweight curriculum progression by enabled mechanics.
- Add regression eval slices per major rule family.

## Done when

- Training runs remain healthy after enabling new mechanics.
- Regressions are diagnosable via added metrics and eval slices.
