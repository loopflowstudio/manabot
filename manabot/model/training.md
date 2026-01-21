# PPO Implementation Guide for manabot

## Algorithm

### Training Loop Structure

```python
def train_iteration(self):
    """Single PPO training iteration"""
    # 1. Collect trajectories
    trajectories = self.collect_trajectories()
    
    # 2. Compute advantages (per player)
    for player_id in range(self.num_players):
        player_advantages = self.compute_player_advantages(
            trajectories.filter(player_id=player_id)
        )
        
    # 3. Policy updates
    for epoch in range(self.num_epochs):
        # Generate minibatches
        for batch in self.get_minibatches():
            # Update policy
            loss = self.compute_ppo_loss(batch)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Early stopping check
            if self.approx_kl > self.target_kl:
                break
```

### Advantage Estimation:`compute_player_advantages`
PPO uses Generalized Advantage Estimation (GAE) to measure how much better an action was than average.

GAE uses two different discount factors:
* γ: The discount factor for the immediate reward in the TD error
* λ: The discount factor for the TD errors

Tuning λ allows you to balance between: 
1. Temporal Difference (TD) learning (rt + γV(st+1)): Low variance but biased by value function errors
2. Full returns (rt + γrt+1 + γ²rt+2 + ...): Unbiased but high variance from accumulating future randomness

The GAE equation combines these approaches:

```math
A^{GAE(γ,λ)}_t = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
```
where δt = rt + γV(st+1) - V(st) is the TD error.

The λ parameter directly controls this tradeoff:
- λ=0: Pure TD learning using rt + γV(st+1). Minimum variance but potentially highly biased by value function errors
- λ=1: Full returns using rt + γrt+1 + γ²rt+2 + .... Maximum variance but unbiased
- λ~=0.95 (recommended): Slightly favors real returns while maintaining some variance reduction from TD estimates

Implementation requirements:
1. Bootstrap values for non-terminal states
2. Handle terminal vs truncated episodes differently
3. Use TD(λ) returns for value targets

### Surrogate Objective: `compute_ppo_loss`
Proximal Policy Optimization (PPO) is a method to allow multiple rounds of policy updates to be computed
on a single batch of trajectories.

One issue under such a scheme is that the variance increases as the policy diverges from the data-generating policy
and the local estimation of state probabilities becomes less accurate. PPO addresses this by setting the gradient to 0 for any parameter that is already sufficiently far from its value under the policy that was used to generate the trajectories. 

We start with the high variance "surrogate objective":
```math
L^{SURROGATE}(θ) = E[\frac{π_θ(a|s)}{π_{θold}(a|s)} A^π(s,a)]
```

PPO introduces clipping to prevent these large changes:

```math
L^{CLIP}(θ) = E[min(
    \frac{π_θ(a|s)}{π_{θold}(a|s)} A^π(s,a),
    clip(\frac{π_θ(a|s)}{π_{θold}(a|s)}, 1-ε, 1+ε) A^π(s,a)
)]
```
- ε is typically set to 0.2, meaning the policy can only change probabilities by ±20%

## Implementation Plan

### Policy Network Architecture and Initialization

Policy initialization is crucial for training stability. Key requirements:

```python
def layer_init(layer: nn.Module, layer_type: str = "hidden") -> nn.Module:
    """Initialize layer with carefully chosen orthogonal initialization scales
    
    Args:
        layer: The neural network layer to initialize
        layer_type: One of "hidden", "policy_final", or "value_final"
    """
    if layer_type == "policy_final":
        std = 0.01  # Small for final policy layer (prevents initial extreme actions)
    elif layer_type == "value_final":
        std = 1.0   # Larger for value layer (allows learning larger value ranges)
    else:  # hidden layers
        std = np.sqrt(2)  # Prevents vanishing gradients in deep networks
        
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer
```

Critical aspects:
- Use orthogonal initialization with sqrt(2) for hidden layers
- Final policy layer uses much smaller initialization (0.01)
- Ensures near-zero mean and small standard deviation initially
- Value network final layer uses std=1.0

### Advantage Computation and Normalization

Accurate advantage computation is essential:

```python
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    dones: torch.Tensor,
    player_ids: torch.Tensor,  # Added to track player perspective
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns from correct player perspectives"""
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    
    # Track active player for proper advantage computation
    active_player = player_ids[-1]
    
    for t in reversed(range(len(rewards))):
        # Only compute advantages for current player's turns
        if player_ids[t] != active_player:
            continue
            
        nextnonterminal = 1.0 - dones[t]
        nextvalues = values[t + 1] if t < len(rewards) - 1 else next_value
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    
    returns = advantages + values
    return advantages, returns

def normalize_advantages(advantages: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
    """Normalize advantages within each minibatch"""
    batch_advantages = advantages[batch_indices]
    return (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
```

Critical aspects:
- Careful handling of episode boundaries
- Proper bootstrapping for non-terminal states
- Advantage normalization per minibatch
- Multi-agent perspective maintained throughout

### Action Masking System

Robust action masking is crucial for Magic's dynamic action space:

```python
class ActionMaskingLayer(nn.Module):
    def __init__(self, action_space_size: int):
        super().__init__()
        self.action_space_size = action_space_size
        self.register_buffer("mask", torch.ones(action_space_size))
    
    def forward(self, logits: torch.Tensor, valid_actions: torch.Tensor) -> torch.Tensor:
        """Apply action mask and handle edge cases"""
        # Set invalid action logits to -inf
        masked_logits = logits.clone()
        masked_logits[~valid_actions] = float('-inf')
        
        # Safety check: Ensure at least one valid action
        if not torch.any(valid_actions):
            raise ValueError("No valid actions available")
            
        return masked_logits
```

Critical aspects:
- Set invalid action logits to -inf
- Handle edge cases (e.g., no valid actions)
- Maintain mask efficiency for large action spaces
- Verify mask correctness before training

### Multi-Agent Policy System

We need to store separate buffers for each player to compute long-run returns.

```python
class MultiAgentBuffer:
    def __init__(self, buffer_size: int, num_players: int):
        self.buffers = {
            player_id: PPOBuffer(buffer_size)
            for player_id in range(num_players)
        }
    
    def add(self, transition: Dict, player_id: int):
        """Add transition to correct player buffer"""
        self.buffers[player_id].add(transition)
    
    def compute_advantages(self):
        """Compute advantages separately for each player"""
        for buffer in self.buffers.values():
            buffer.compute_player_advantages()
```

For now though, we are just going to use a single policy for both players.

## Validation and Testing

### Policy
* All probability ratios π_new/π_old must be valid (no NaNs)
* KL divergence between old/new policies stays below target
* At least one valid action always available after masking
* Logits for invalid actions properly set to -inf
* Action sampling never selects invalid actions

### Advantage Estimation
* Advantages only computed within same player's turns
* Bootstrap values only used for non-terminal states
* Advantage normalization preserves sign
* GAE computation properly handles episode boundaries
* Value estimates exist for all non-terminal states

### State Management
* Trajectories maintain correct temporal ordering
* Player perspectives remain consistent throughout updates
* Action masks correctly reflect current game state
* Reward attribution matches active player
* Episode termination properly distinguished from truncation

### Hyperparameter Tuning

    Probably not going to do any systematic sweeps to start.
    See infra/hypers.py for the current hyperparameters.

#### Diagnostics
 * If approx_kl skyrockets early, the learning rate or clip range might be too large.
 * If approx_kl stays near zero, maybe the policy isn’t learning aggressively enough.
 * If entropy plummets to near-zero quickly, the model may be collapsing to a single set of actions (overfitting).
 * If value loss is very large or consistently grows, check for improper reward scaling or network initialization 

 ### Training Runs

Right now I will be training for 1min-24hr runs on my Mac.
Will move to doing a 1-7day runs on a remote GPU once that is working well.
After that is working I will investigate in some basic distributed training infrastructure.

## Future Projects

### Separate Value Network 
 * The original PPO paper uses a separate, wider value network.

### Neural Network Architectures
 * There's a lot of room for experimentation here.
    LSTMs or Attention seem like good candidates.

### Value Loss Clipping
  * Original PPO paper recommends against clipping

### Advantage Normalization
 * Per-minibatch more mathematically precise
 
hese decisions should be resolved based on empirical testing with the manabot environment.

## References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms" (2017)
2. Schulman, J., et al. "High-dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
3. "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO" (2020)
4. "The 37 Implementation Details of Proximal Policy Optimization" (2022)

