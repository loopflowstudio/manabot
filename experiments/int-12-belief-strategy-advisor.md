# INT-12: Belief-Conditioned Strategy Advisor

## Result

The advice-v1 vertical slice resolves one canonical committed decision through
both Game's live-session capability and its Study fork capability. Given the
same viewer, authored belief scenarios, advisor identity, compute profile, and
paired seed plan, both routes return the same 13,804 response bytes with
SHA-256 `237c91b9c16aeaa41de48c3af990b807233574716ae5272f5b4d69d9d556dc6f`.

The compatible-deal prior and authored Bolt-heavy range produce a maximum
absolute policy-probability delta of 0.125. This is one deterministic
engineering fixture with random terminal leaves. It is not a method, strength,
rating, admission, or service-SLO claim.

## Reproduce

From the repository root:

```bash
uv run --extra dev experiments/runners/run_belief_strategy_advisor.py --verify-fixture --write-measurement
```

Verification regenerates the canonical request, response, and schema in
memory; checks the committed fixture byte for byte; requires a nonzero strategy
delta; proves the live and Study response bytes are identical; and measures at
least 20 warmups plus 128 calls for each provider path. Fixture or schema
maintenance instead requires the explicit `--update-fixture` flag.

## Declared profile

- Planner: determinized PUCT with uniform-random terminal evaluation.
- Compute: eight traversals per scenario, two sampled worlds per scenario,
  `c_puct=1.5`, and an 80-step cap.
- Seed: root 197 with paired inverse-CDF draws shared across scenarios.
- Realized comparison cost: 16 simulations, four materialized worlds, 20 tree
  nodes, and 14 cap hits.
- Cached latency: 0.984 ms p50 and 1.724 ms p95.
- Fresh recomputation latency: 16.981 s p50 and 27.435 s p95.
- Peak process RSS: 2,810,789,888 bytes.

The complete hardware fingerprint, component timings, sample counts, and
declared method identity are retained in
[`data/int-12-belief-strategy-advisor-measurement-v1.json`](data/int-12-belief-strategy-advisor-measurement-v1.json).

## Boundary

The public response contains the authoritative semantic offer order, normalized
belief receipts, policy visits and probabilities, supported value/robustness/
uncertainty quantities, realized aggregate compute, and exact provenance. It
does not contain belief weights, materialized hands, world indexes, branch
receipts, RNG tapes, actual hidden-query truth, or a private root digest.
Unavailable checkpoints, manifests, identities, ranges, and authority roots
return closed typed states without partial policy evidence.
