manabot Neural Architecture: Revised Design (Small-Scale Bias)
Core Design Philosophy
manabot views Magic: The Gathering as a set of interacting objects—players, cards, permanents—where every piece may affect every other. Our design is guided by three principles:

Three Design Prioirties:

 * Simplicity and Symmetry
 * Direct Access
 * Global context


Architecture Overview
The architecture consists of two main stages:

Object Encoding and Global Attention
Action Evaluation with Direct Target Incorporation

Game State Processing
Each type of game object (players, cards, permanents) is processed by its own encoder (a small MLP) to produce compataible embeddings in a unified tensor.
All objects are added with a learned perspective vector. When added with a +1 (for agent) or –1 (for opponent).
A standard multi-head attention layer is applied to all the objects.

NOTE: We are not currently using the global state of the game.
RISK: Attention layer is O(n^2) in the number of objects which could potentially reach 1000s. 

Implementation Snippet:

python
Copy
class GameStateProcessor(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.perspective = nn.Parameter(torch.randn(embedding_dim) / embedding_dim**0.5)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(embedding_dim, 4 * embedding_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(4 * embedding_dim, embedding_dim))
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, objects: torch.Tensor, controller_ids: torch.Tensor,
                agent_id: torch.Tensor) -> torch.Tensor:
        # Add perspective: agent gets +v, opponent gets -v.
        is_agent = (controller_ids == agent_id.unsqueeze(-1))
        perspective_scale = torch.where(is_agent, 1.0, -1.0).unsqueeze(-1)
        objects = objects + perspective_scale * self.perspective
        # Global attention over all objects.
        attn_out, _ = self.attn(objects, objects, objects)
        x = self.norm1(objects + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

Second Stage: Action Processing
Action features are projected using a small MLP (the action_projector). 

CURRENT: I think we currently just do not do anything with the focus objects.

PLAN:
We can should concatenate the embedded action with (pre_attention_focus_object, post_attention_focus_object)

RISK: We are not 

Policy and Value Heads:
We use simple feed-forward networks to compute policy logits and state value estimates. 


No Explicit Target Attention:
We do not yet implement a secondary attention mechanism to fuse target information. Instead, target information (if available) would be concatenated directly. This may need refinement if targeting proves crucial.
Fixed Capacity:
The design uses fixed dimensions (with padding/masking for smaller states), which works well at a small scale but might need rethinking for dynamic game states later.
Implementation Snippet:

python
Copy
class ActorCritic(nn.Module):
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gather and encode all game objects.
        objects, controller_ids = self._gather_object_embeddings(obs)
        agent_id = obs["agent_player"][:, 0].long()
        processed_state = self.state_processor(objects, controller_ids, agent_id)
        
        # Process actions: use all but the last column as action features.
        act_feats = obs["actions"][:, :, :-1]
        B, A, _ = act_feats.shape
        action_embeddings = []
        for i in range(A):
            single_act = act_feats[:, i, :]
            act_proj = self.action_projector(single_act)
            # (Placeholder) Future work: if target indices are provided, lookup from processed_state.
            action_embeddings.append(act_proj)
        action_embeddings = torch.stack(action_embeddings, dim=1)
        
        logits = self.policy_head(action_embeddings)
        value = self.value_head(action_embeddings.mean(dim=1)).squeeze(-1)
        return logits, value
Implementation Details
Object Encoding
Each object type is encoded using a dedicated MLP. This works well on our current small-scale data:

Players, Cards, Permanents:
Each type’s features are passed through its respective encoder and later concatenated into a unified tensor for attention.
Attention Implementation
We use standard multi-head attention (quadratic in the number of objects) because:

For small game states (50–100 objects), computation and memory usage are well within acceptable limits.
This implementation is straightforward and well-optimized in PyTorch.
Future Work: If we scale to thousands of objects, we will consider sparse or hierarchical attention.
Action Target Handling
We currently assume that explicit target indices are not provided. The design allows for direct tensor indexing (using GPU-optimized operations) to extract target information from the processed state. In our present iteration, this step is a placeholder to be refined as we gather more evidence on its impact.

Testing Focus
Our testing strategy emphasizes:

Attention Coverage:
Verifying that every object correctly attends to every other.
Testing with varying game state sizes (within the small-scale regime).
Target Processing:
Validating correct (or placeholder) behavior for target lookup.
Ensuring that masking and padding for invalid objects work correctly.
Computational Efficiency:
Benchmarking throughput on typical game sizes.
Monitoring memory usage to ensure learnability on small-scale inputs.
Summary and Next Steps
Current Implementation Strengths:

A simple, symmetric design that enforces weight reuse.
A modular architecture with separate object encoding, global attention, and action processing stages.
Stability on small-scale inputs (50–100 objects) with standard multi-head attention and basic MLPs.
Current Limitations:

No explicit target attention mechanism; target incorporation is handled by direct concatenation (or left as a placeholder).
Dense attention scaling may become a bottleneck if game states scale up significantly.
Further experimentation is needed to determine the precise contribution of residual paths and the ideal fusion strategy for action features and target information.
Bias Toward Learnability on a Small Scale:
The design intentionally avoids complex mechanisms (e.g., secondary attention or sophisticated gating) to ensure that the system is learnable and debug-friendly on small game states. We plan to add additional complexity only once we have a stable and well-understood baseline.

By focusing on these essentials, we create a solid foundation that can be incrementally enhanced while ensuring that early iterations are both interpretable and stable.

This updated design document reflects both the current implementation and our strategy to ensure learnability on a small scale. It is explicit about what works now, what is a placeholder for future improvement, and how we plan to scale if needed. Happy coding and iterative experimentation!







## Where the code lives

The policy is split by what depends on an array framework:

- [`manabot/policy/`](../policy/README.md): observation admission, bucket
  trimming, host preparation (definition rows, record vocabulary, legal
  actions, packed GRU schedules, shared-effect routes, column compaction) and
  the checkpoint container. NumPy only.
- [`manabot/backends/torch/`](../backends/README.md#the-torch-backend):
  `ActorCritic` (`policy.py`) and its encoders, the forward over a prepared
  batch. Other frameworks sit beside it under
  [`manabot/backends/`](../backends/README.md).
- This directory: architecture receipts ([`architecture.py`](architecture.py)),
  the checkpoint catalog ([`catalog.py`](catalog.py)) and world binding
  ([`world.py`](world.py)).

## Observation admission

A policy forward reads only a `Prepared` batch. The policy's `prepare`
validates a batch of NumPy observation buffers once, on the host, and builds
every index the forward reads; `forward`, `forward_distribution`,
`get_action_and_value` and `get_value` then do no validation and no lookups
of their own:

```python
from manabot.backends.torch.arrays import on_device

prepared = agent.prepare(obs)                    # NumPy: checked, trimmed, indexed
logits, value = agent(on_device(prepared, "cpu"))  # logits [B,A], value [B]
```

Admission (`manabot.policy.admission.check`) reads the observation, not the
weights: shapes, binary validity, definition and vocabulary references, action
focus and continuation references, permanent links, fact and record programs,
shared-effect partitions, naming spans, recent events, belief rows, the
legal-action mask and, when the value aggregation needs one, a valid object in
every row. Malformed input raises `ValueError` naming the check (a malformed
action mask is `NumericalError` `invalid_action_mask`); nothing is truncated or
repaired. The prepared batch records whether every row has a legal action, so
`get_action_and_value` rejects an empty row with `No valid actions available`
without reading the batch again. The MLX policy runs the same checks.

An observation dictionary, or a prepared batch still in NumPy, passed to a
forward raises `TypeError`. A prepared batch belongs to the `ModelConfig` and
observation configuration that prepared it, not to one instance: a restored,
target or bootstrap copy reads it, and a model with another configuration
raises `ValueError`. Its arrays are read-only; assembling a `Prepared` by hand
is outside the contract.

Preparation also chooses the batch's data-dependent structure: legal action
rows, bucket sizes, the object and fact columns in use, packed card programs,
live-token offsets and shared-effect CSR routes. Those reads choose the work;
they are not validation, and they happen on the host so the device forward
reads nothing back.

Each observation path prepares once per batch:

- Collection and search serving prepare inside `PolicyInference.prepare`
  (below).
- Learning wraps each rollout batch in `AdmittedRollout`
  ([`rollout.py`](../backends/torch/rollout.py)): it checks the bootstrap
  rows and then the rows an update trains, in minibatch-sized chunks; rows the
  advantage filter drops are never checked, and epochs prepare admitted rows
  again without repeating the checks.
- Other callers use `agent.prepare(obs)` and `on_device`, or
  `agent.evaluate(agent.prepare(obs))` for NumPy logits and values.

## Inference preparation

Ordinary seat-routed collection uses `PolicyInference`
([`policy.py`](../backends/torch/policy.py)) for one unchanged
behavior-policy interval. Its `prepare` takes NumPy observation buffers,
prepares them and moves the batch to the model's device; the policy lazily
encodes the semantic catalog once and reuses it for the interval. A batch it
returns must not be mutated or used after the scope closes. Normal parameter
updates, tensor replacement, model-mode/configuration changes and
gradient-enabled use are rejected. Unsafe mutation through `.data` aliases is
outside this contract. A fresh scope follows every collection/update
boundary. The cached catalog is never part of the state dict or a checkpoint.

## Capacity identity and accounting

`ModelConfig.hidden_dim`, positive `attention_layers`, optional
`attention_feedforward_dim`, and `num_attention_heads` own the capacity configuration. The builder
`manabot.recipes.builders.with_capacity` reuses those fields; they introduce no second model schema. Heads must divide width when
attention is enabled. Unset feedforward expansion is head-count times width,
so changing heads can change parameter counts as well as attention semantics.
The default ladder fixes four heads and compares 64/1, 64/2 and 128/2.
Explicit `include_ataraxos=True` adds 384/8 with feedforward width 1,536;
[the scale probe](../../docs/experiments.md#freeze-step-targets)
measures that model without changing the scientific cohort.

`ActorCritic.architecture_receipt()` passes its named parameter sizes to
`manabot.model.architecture.architecture_receipt`, which derives a versioned
identity and total/trainable counts by the actual top-level parameter owner.
It counts shared parameters once and excludes buffers. The receipt consumes no
RNG and does not change initialization, weights or forward equations. Counts
depend on semantic inputs, observation dimensions, output representation and
optional modules; the historical 138,434 figure is not a universal baseline.

The identity hashes resolved ModelConfig and observation settings under layout
version 1. Equation/ownership changes require a new layout version even for equal
shapes. World/content bindings, source/runtime digests and exact checkpoint bytes
remain separate authorities. Run identities record the architecture hash; ordinary
PPO and BC/regime exports record the receipt. Reload validates present receipts,
including same-shaped metadata discrepancies. Old checkpoints without receipts
retain strict world/input/state-dict admission; no weight transfer or metadata
reinterpretation is performed. Trainability records export-time flags and is
not an optimizer-exposure counter. See [bounded calibration](../../docs/experiments.md#freeze-step-targets).
