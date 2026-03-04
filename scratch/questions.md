# Open questions / assumptions

1. **C++ throughput baseline re-measurement**
   - Assumption used: proceed with Rust cutover despite no fresh local C++ re-run.
   - Reason: sandbox environment does not include CMake (`cmake: command not found`), so the legacy C++ backend cannot be rebuilt for an apples-to-apples run.
   - Mitigation: Rust throughput run is documented in `scratch/benchmark.md` and can be compared to any previously captured C++ baseline externally.

2. **Training smoke test**
   - Attempted command:
     `python manabot/model/train.py train.total_timesteps=128 train.num_envs=2 train.num_steps=8 train.update_epochs=1 train.num_minibatches=1 experiment.wandb=false experiment.profiler_enabled=false agent.attention_on=false`
   - Blocker: local environment is missing `psutil` and this sandbox has no network access to install new dependencies.
   - Impact: cutover verification includes cargo/pytest/enum smoke and benchmark doc, but not a completed PPO smoke run in this session.
