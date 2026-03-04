# Open questions / assumptions

1. **Step count divergence between C++ and Rust**
   - C++ produces 35,805 steps for 500 games; Rust produces 40,500 steps (same seed, same deck, same policy).
   - Both complete all 500 games. The difference is in action generation or priority handling, not game completion.
   - Not blocking cutover — Rust behavior defines correct going forward.

2. **Training smoke test**
   - psutil now installed. Can be re-run:
     `python manabot/model/train.py train.total_timesteps=128 train.num_envs=2 train.num_steps=8 train.update_epochs=1 train.num_minibatches=1 experiment.wandb=false experiment.profiler_enabled=false agent.attention_on=false`
