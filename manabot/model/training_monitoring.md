# Enhanced Monitoring and Debugging Plan for MTG RL PPO Bot

This document combines general research findings with specific observations from a full codebase review. It is designed to serve as a reference guide for a scrappy entrepreneur looking to iterate quickly on a Magic: The Gathering (MTG) reinforcement learning (RL) PPO bot that now includes an attention mechanism. The guide focuses on key opportunities for monitoring, debugging, and ensuring architectural soundness.

---

## Table of Contents

1. [Overview of Monitoring Goals](#overview-of-monitoring-goals)
2. [Detailed Component Monitoring & Debugging Targets](#detailed-component-monitoring--debugging-targets)
   - [Agent (manabot/ppo/agent.py)](#agent-manabotppoagentpy)
   - [Environment & Observation (manabot/env/ and manabot/env/observation.py)](#environment--observation-manabotenv--and-manabotenvobservationpy)
   - [Trainer (manabot/ppo/trainer.py)](#trainer-manabotppotrainerpy)
   - [Profiler (manabot/infra/profiler.py)](#profiler-manabotinfraprofilerpy)
   - [Simulation & Evaluation (manabot/sim/)](#simulation--evaluation-manabotsim)
3. [Implementation Roadmap](#implementation-roadmap)
4. [Example Code Snippets](#example-code-snippets)
   - [Gradient Norm Logging](#gradient-norm-logging)
   - [Attention Assertion](#attention-assertion)
5. [Questions for Clarification](#questions-for-clarification)
6. [Summary and Next Steps](#summary-and-next-steps)

---

## Overview of Monitoring Goals

- **Gradient and Weight Flow:**
  - Monitor per-layer gradient norms and weight update magnitudes during training.
  - Detect exploding/vanishing gradients early—especially critical after integrating the attention layer.

- **Attention Verification:**
  - Ensure the `key_padding_mask` correctly zeros out masked positions.
  - Verify that altering padded token values does not affect output (i.e. no leakage through the mask).

- **Observation & Encoding:**
  - Validate that the observation encoder produces consistent shapes, valid masks, and a correct object-to-index mapping.
  - Log summary statistics (e.g. total valid objects per type) to catch unexpected changes.

- **Policy & Action Outputs:**
  - Track action probability distributions, entropy, and ensure that invalid actions are masked (logits set to –1e8).
  - Check consistency between the direct forward pass and the action-sampling interface.

- **Performance & System Metrics:**
  - Leverage the hierarchical profiler to monitor time breakdown across submodules (rollout, gradient computation, environment step).
  - Log system metrics (memory, CPU/GPU usage) via wandb or local logging.

- **Simulation & Evaluation:**
  - Record behavioral metrics (win rates, game lengths, action distributions) during simulation.
  - Build a head-to-head evaluation (including basic ELO calculation) to compare different model versions.

---

## Detailed Component Monitoring & Debugging Targets

### Agent (manabot/ppo/agent.py)

- **Gradient Norms & Weight Updates:**
  - Insert logging in the training loop (e.g., in `_optimize_step`) to record:
    - Total gradient norm per update.
    - Per-layer gradient norms for key modules (embeddings, attention, policy head).

- **Attention Module:**
  - In `GameObjectAttention.forward`, verify that outputs for masked positions are exactly zero.
  - Add tests that compare outputs when padded token embeddings are randomized, ensuring that changes do not propagate.

- **Action Focus Processing:**
  - Monitor outputs of `_add_focus` by logging shapes and nonzero statistics.
  - Confirm that positions with focus index `-1` yield zero embeddings.

---

### Environment & Observation (manabot/env/ and manabot/env/observation.py)

- **Observation Validation:**
  - Enhance the `_validate_obs` function in Trainer to log discrepancies in expected keys/shapes.
  - Log summary statistics from validity masks (e.g., total valid counts per object type) to detect anomalies.

- **Object-to-Index Mapping:**
  - Periodically log a summary or histogram of indices assigned by the observation encoder.
  - Add unit tests to verify that the mapping is consistent across resets.

---

### Trainer (manabot/ppo/trainer.py)

- **Loss Components & Early Stopping:**
  - Log detailed breakdowns of policy loss, value loss, entropy, and gradient norms.
  - Monitor KL divergence (`approx_kl`) and clipping fraction; set alerts if these exceed thresholds.

- **Buffer and Advantage Computation:**
  - Log basic statistics (min, max, mean) of computed advantages.
  - Insert assertions to ensure that GAE does not produce NaN or Inf values.

- **System Metrics:**
  - Continue logging system metrics (memory, CPU, GPU) via `_log_system_metrics`.
  - Optionally add additional wandb log calls for new metrics during troubleshooting.

---

### Profiler (manabot/infra/profiler.py)

- **Hierarchical Timing:**
  - Verify that each nested timer’s output is as expected.
  - Consider a “live monitor” mode that reports running nodes’ effective times to identify bottlenecks.

---

### Simulation & Evaluation (manabot/sim/)

- **Behavioral Metrics:**
  - Log per-game statistics (number of turns, win rates, action distributions) to capture behavioral trends.
  - Enhance simulation code to integrate additional wandb logs for detailed game outcome summaries.

- **Comparison Tool:**
  - Develop a head-to-head evaluation module that runs controlled experiments between model versions.
  - Log win rates, average game lengths, and compute a basic ELO as diagnostic metrics.

---

## Implementation Roadmap

### Step 1: Instrument Training (Trainer & Agent)
- **Insert Gradient Logging:**
  - Add code in `_optimize_step` to record total and per-layer gradient norms.
- **Extend Loss Logging:**
  - Ensure wandb logs include all loss components and gradient norm values.
- **Attention Checks:**
  - In `GameObjectAttention.forward`, insert an assertion ensuring masked outputs are zero.

### Step 2: Validate Observation Encoder
- **Log Validity Mask Sums:**
  - After encoding, log the sum of valid flags per object type.
- **Unit Tests:**
  - Expand tests in `test_observation.py` to verify validity mask statistics and the consistency of the object-to-index mapping.

### Step 3: Enhance Simulation Logging
- **Behavior Logging:**
  - Integrate additional logging into simulation functions (in `manabot/sim/player.py` and `manabot/sim/sim.py`) to record action distributions and game outcomes.
- **Action Distribution & Confidence:**
  - For `ModelPlayer`, record softmax probability statistics (mean confidence, entropy) and log these metrics.

### Step 4: Monitor Performance & System Metrics
- **Profiler Enhancements:**
  - Optionally add a “live dashboard” that prints current timing for critical nodes.
- **System Resource Logging:**
  - Verify that `_log_system_metrics` is called regularly; adjust frequency if needed during debugging.

### Step 5: Update and Expand Tests
- **Expand Unit Tests:**
  - Add tests for invalid observations and ensure `_validate_obs` catches these issues.
  - Test the attention module separately with controlled inputs to ensure proper masking.

---

## Example Code Snippets

### Gradient Norm Logging

Insert the following code snippet into the training loop (e.g., in `Trainer._optimize_step`):

```python
# After loss.backward()
total_grad_norm = 0.0
for name, param in self.agent.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.detach().data.norm(2).item()
        total_grad_norm += grad_norm ** 2
        self.logger.debug(f"Grad norm [{name}]: {grad_norm:.4f}")
total_grad_norm = total_grad_norm ** 0.5
self.logger.info(f"Total gradient norm: {total_grad_norm:.4f}")
Attention Assertion
In GameObjectAttention.forward, add an assertion after masking:

python
Copy
def forward(self, objects: torch.Tensor, is_agent: torch.Tensor, key_padding_mask: torch.BoolTensor) -> torch.Tensor:
    # ... [existing attention code] ...
    post_norm = self.norm2(x + mlp_out)  # [B, total_objs, embedding_dim]
    # ZERO OUT outputs for masked (invalid) positions.
    mask = (~key_padding_mask).unsqueeze(-1).float()  # valid positions: 1.0; invalid: 0.0
    post_norm = post_norm * mask
    # Extra assertion for debugging: masked positions must be zero.
    assert torch.all(post_norm[mask == 0] == 0), "Attention output leakage: masked positions are nonzero"
    return post_norm
