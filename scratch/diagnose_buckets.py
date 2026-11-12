"""Which gradients differ relatively, and how many input shapes buckets leave."""
import sys
import torch
sys.argv = [sys.argv[0], sys.argv[1], "/dev/null"]
source = open("scratch/measure_buckets.py").read().split("def main()")[0]
exec(source)
for size in ("w64", "w160l8"):
    default, bucketed = agents(size)
    enc = default.observation_space.encoder
    tuples = set()
    for path in sorted((E / f"captures/{size}/collection").glob("*.npz")):
        tuples.add(tuple(bucket_sizes(load_obs(path), enc, objects=True).sizes.values()))
    learner_tuples = set()
    worst = []
    for path in sorted((E / f"captures/{size}/learner").glob("*.pt")):
        capture = torch.load(path, weights_only=False)
        obs, actions = capture["obs"], capture["actions"].long()
        learner_tuples.add(tuple(bucket_sizes(obs, enc, objects=True).sizes.values()))
        for agent in (default, bucketed):
            agent.zero_grad(set_to_none=True)
            logits, values = agent(obs)
            chosen = torch.log_softmax(logits, -1).gather(1, actions[:, None]).squeeze(1)
            (-(chosen * capture["advantages"]).mean() + ((values - capture["value_targets"]) ** 2).mean()).backward()
        for (name, a), b in zip(default.named_parameters(), bucketed.parameters()):
            if a.grad is None:
                continue
            diff = float((a.grad - b.grad).abs().max()); scale = float(a.grad.abs().max())
            norm_rel = float((a.grad - b.grad).norm() / a.grad.norm().clamp_min(1e-30))
            worst.append((diff / scale if scale else 0, norm_rel, diff, scale, name, path.name))
    worst.sort(reverse=True)
    print(size, "collection input signatures", len(tuples), "learner", len(learner_tuples))
    for row in worst[:6]:
        print("  rel %.3g norm_rel %.3g abs %.3g scale %.3g %s %s" % row)
    print("  max norm_rel", max(w[1] for w in worst))
