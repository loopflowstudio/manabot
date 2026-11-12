"""Pooled vs tables on captured TLA learner minibatches, CPU: runs, finite, batch-independent."""
import json, sys
from pathlib import Path
import torch
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic

torch.set_num_threads(1)
E = Path(sys.argv[1])
regime = json.loads(Path(sys.argv[2]).read_text())
space = ObservationSpace(ObservationConfig.model_validate(regime["observation"]))
base = {**regime["agent"], "hidden_dim": 64, "attention_layers": 2, "num_attention_heads": 4}
for arm in ("tables", "pooled"):
    torch.manual_seed(1)
    agent = ActorCritic(space, ModelConfig.model_validate({**base, "token_stream": arm}))
    print(arm, sum(p.numel() for p in agent.parameters()))
    for path in sorted((E / "learner").glob("*.pt"))[:3]:
        obs = torch.load(path, weights_only=False)["obs"]
        agent.train()
        logits, value = agent.forward_distribution(obs)
        legal = obs["actions_valid"].bool()
        loss = -torch.log_softmax(logits, -1)[legal].mean() + value.square().mean()
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in agent.parameters() if p.grad is not None)
        agent.zero_grad()
        agent.eval()
        with torch.no_grad():
            full, v_full = agent(obs)
            head = {k: v[:3] for k, v in obs.items()}
            part, v_part = agent(head)
        l3 = legal[:3]
        print(" ", path.name, obs["actions_valid"].shape[0], float(loss), float((full[:3][l3] - part[l3]).abs().max()), float((v_full[:3]-v_part).abs().max()))
