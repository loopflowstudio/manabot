import sys, torch
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentSpec
from manabot.model import Agent
from manabot.verify.action_cases import curriculum
torch.set_num_threads(1)
cases = curriculum(); fit = [c for c in cases if c.split == "fit"]
batch = {k: torch.cat([c.observation[k] for c in fit]) for k in fit[0].observation}
labels = torch.tensor([c.preferred for c in fit])
for seed in map(int, sys.argv[1:]):
    torch.manual_seed(seed)
    agent = Agent(ObservationSpace(), AgentSpec(hidden_dim=32, num_attention_heads=2, attention_feedforward_dim=64, policy_film=True))
    opt = torch.optim.Adam(agent.parameters(), lr=0.003)
    for _ in range(240):
        loss = torch.nn.functional.cross_entropy(agent(batch)[0], labels); opt.zero_grad(); loss.backward(); opt.step()
    agent.eval()
    out = []
    for c in cases:
        if c.family != "learn" or c.variant != 1: continue
        o = c.observation
        swapped = {k: v.clone() for k, v in o.items()}
        swapped["semantic_cards"][swapped["semantic_cards"] == 3] = 2
        with torch.no_grad():
            p = agent(o)[0].softmax(-1)[0, c.preferred].item(); q = agent(swapped)[0].softmax(-1)[0, c.preferred].item()
        out.append(f"p{c.preference}: {p:.2f} -> swapped {q:.2f}")
    print(seed, out)
