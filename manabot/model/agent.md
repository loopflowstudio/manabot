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
class Agent(nn.Module):
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






