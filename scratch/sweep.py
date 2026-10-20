"""30-seed sweep of the learnability capacity fit with the gate's Config."""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import torch

def one(args):
    seed, kw = args
    variant = kw.pop('_variant', None)
    if variant:
        import sys; sys.path.insert(0, 'scratch'); import proto; proto.install(variant)
    torch.set_num_threads(1)
    from manabot.env import ObservationSpace
    from manabot.infra.hypers import AgentSpec
    from manabot.model import Agent
    from manabot.verify.action_cases import curriculum
    from manabot.verify.learnability.capacity import _score
    from manabot.verify.learnability.schema import Config
    cfg = Config()
    cases = curriculum()
    if kw.pop('_bug', False):
        from manabot.verify.action_cases import identity_removed
        for c in cases:
            if c.family in ('attack','mode','ability'): c.observation = identity_removed(c)
    fit = [c for c in cases if c.split == "fit"]
    batch = {k: torch.cat([c.observation[k] for c in fit]) for k in fit[0].observation}
    labels = torch.tensor([c.preferred for c in fit])
    spec = AgentSpec(hidden_dim=32, num_attention_heads=2, attention_feedforward_dim=64, **kw)
    torch.manual_seed(seed)
    agent = Agent(ObservationSpace(), spec)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
    agent.train()
    for _ in range(cfg.updates):
        logits, _ = agent(batch)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    agent.eval()
    fails = {}
    for c in cases:
        s = _score(agent, c, seed, 0.0, cfg)
        if not s.passed:
            fails.setdefault(c.family, []).append(f"{c.split[0]}{c.variant}p{c.preference}:{s.trained:.2f}")
    return seed, float(loss), fails

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1-30")
    p.add_argument("--spec", default="{}")
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    kw = json.loads(a.spec)
    t = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, [(s, kw) for s in range(lo, hi + 1)]))
    bad = 0
    for seed, loss, fails in res:
        if fails: bad += 1
        print(seed, f"loss={loss:.4f}", fails if fails else "ok")
    print(f"failing seeds: {bad}/{len(res)}  ({time.time()-t:.0f}s)")
