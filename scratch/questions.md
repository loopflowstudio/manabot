# Assumptions (reversible)
- The optional extra is named `mlx`; API names (`convert`, `parity`, `load_mlx_policy`, `MLXLearner`) set in manabot/model/mlx.md.
- experiments/cuda_precision.md (MTG-317) is on origin/main but not on this branch's base; the report links it. Rebase before landing.
- Prediction numbers in the committed paragraph (1.3x / 1.8x) were written by the agent before any measurement.
- Gradient-cosine bar clarified before timing: tensors below 1e-6 of the global gradient norm in both frameworks count as zero
  (policy-score bias and attention key biases have analytically zero gradients). Stated in the report and parity docstring.
- Learning keeps fixed padded shapes in both arms (production regime has bucketed_shapes=False); MLX inference compacts like torch.
- Published evidence: ledgers, summary, charts, capture metadata, and a parity subset (initial.pt, 8 minibatches, 16 calls);
  the full 12 GB capture stays on the host and is lost when it is released.
- Publishing (manabot.data publish) runs on the laptop: an upload, not MLX/torch compute; R2 creds from Doppler manabot/prd.
