"""State-conditioned choices over real authored-game decisions.

Collect native choices from the selected world (choice_corpus), replay each at
two public life totals for the acting viewer, and ask one shared policy to
prefer group A at life 5 and group B at life 35. Reports per-seed counts of
cases fit (>=0.9 both levels) and cases stuck symmetric (both in 0.4..0.6).
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch

CACHE = Path("scratch/real_states_cases.pt")

def build(rounds: int) -> None:
    import managym
    from manabot.verify.action_cases import encode
    from manabot.verify.choice_corpus import PACK, _signatures, collect_choices
    rows = []
    for r in range(rounds):
        cases, _, _ = collect_choices(Path(f"/tmp/mtg154-corpus/r{r}"), r, time.perf_counter() + 600)
        for case in cases:
            configs = [managym.authored_deck_setup(PACK, k) for k in ("ur_lessons", "gw_allies")]
            if case.swapped: configs.reverse()
            env = managym.Env(seed=case.seed, skip_trivial=False)
            env.reset(configs)
            for i in case.trace: env.step(i)
            assert env.state_digest() == case.digest
            acting = env.current_agent_index()
            pair = []
            for life in (5, 35):
                b = env.clone_env(); b.scenario_set_life(acting, life)
                raw = b.observation_for_player(acting)
                if _signatures(raw) != case.signatures: break
                pair.append(encode(b, raw))
            if len(pair) == 2:
                rows.append({"name": case.name, "family": case.family, "groups": case.groups, "obs": pair})
    torch.save(rows, CACHE)
    print(f"{len(rows)} real cases", {f: sum(r['family']==f for r in rows) for f in sorted({r['family'] for r in rows})})

def one(args):
    seed, kw = args
    torch.set_num_threads(1)
    variant = kw.pop("_variant", None)
    if variant:
        sys.path.insert(0, "scratch"); import proto; proto.install(variant)
    from manabot.env import ObservationSpace
    from manabot.infra.hypers import AgentSpec
    from manabot.model import Agent
    rows = torch.load(CACHE, weights_only=False)
    obs = [o for r in rows for o in r["obs"]]
    keys = obs[0].keys()
    width = max(o["actions"].shape[1] for o in obs)
    batch = {k: torch.cat([o[k] for o in obs]) for k in keys}
    selected = torch.zeros_like(batch["actions_valid"], dtype=torch.bool); pair = selected.clone()
    for i, r in enumerate(rows):
        for lvl in (0, 1):
            row = 2 * i + lvl
            selected[row, r["groups"][lvl]] = True
            pair[row, r["groups"][0] + r["groups"][1]] = True
    spec = AgentSpec(**kw)
    torch.manual_seed(seed)
    agent = Agent(ObservationSpace(), spec)
    opt = torch.optim.Adam(agent.parameters(), lr=0.003)
    agent.train()
    for _ in range(240):
        logits, _ = agent(batch)
        loss = (logits.masked_fill(~pair, -torch.inf).logsumexp(-1) - logits.masked_fill(~selected, -torch.inf).logsumexp(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    agent.eval()
    with torch.no_grad():
        logits, _ = agent(batch)
    p = (logits.masked_fill(~selected, -torch.inf).logsumexp(-1) - logits.masked_fill(~pair, -torch.inf).logsumexp(-1)).exp().view(-1, 2)
    fit = int(((p >= 0.9).all(1)).sum()); stuck = int((((p - 0.5).abs() < 0.1).all(1)).sum())
    return seed, len(rows), fit, stuck, float(loss)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--build", type=int, default=0)
    a.add_argument("--seeds", default="1-30")
    a.add_argument("--spec", default='{"hidden_dim":32,"num_attention_heads":2,"attention_feedforward_dim":64}')
    a.add_argument("--jobs", type=int, default=7)
    args = a.parse_args()
    if args.build: build(args.build); raise SystemExit
    lo, hi = map(int, args.seeds.split("-"))
    kw = json.loads(args.spec)
    with ProcessPoolExecutor(args.jobs) as ex:
        res = list(ex.map(one, [(s, dict(kw)) for s in range(lo, hi + 1)]))
    for r in res: print(*r)
    n = res[0][1]
    print(f"cases={n} mean_fit={sum(r[2] for r in res)/len(res):.1f} mean_stuck={sum(r[3] for r in res)/len(res):.2f} seeds_with_stuck={sum(r[3]>0 for r in res)}/{len(res)} seeds_all_fit={sum(r[2]==n for r in res)}/{len(res)}")
