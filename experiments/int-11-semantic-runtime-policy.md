# INT-11: Learned Semantic Runtime Policy

**World:** w2  
**Run:** 2026-07-18  
**Checked data:**
[`data/int-11-semantic-runtime-policy-v1.json`](data/int-11-semantic-runtime-policy-v1.json)

## Question

Can a small learned manabot consume viewer-safe `ExperienceFrame` facts,
typed ability-program structure, and authoritative `InteractionOffer` values,
then emit normal structured Commands in priority, targeting, and combat while
transferring to held-out identities or known-operation compositions?

## One-command reproduction

```bash
uv run experiments/runners/run_semantic_runtime_policy.py \
  --out-dir .runs/int-11-semantic-runtime-policy-v1
```

The command regenerates 24 authoritative examples, trains semantic,
identity-only, and fixed structure-shuffled arms at seeds 1101/1102/1103,
loads and verifies all nine content-addressed checkpoints, evaluates both
holdouts and the 35-target/64-combat-branch frontiers, benchmarks inference,
plays the paired w2 matchup, independently replays evaluation and arena
Commands, and writes a versioned manifest.

## Result

**The runnable policy and artifact integrity pass; evidence for intact semantic
structure is null or ambiguous.** All 144 decoded Commands were legal, all 108
evaluation rows and 36 arena Commands replayed with exact source/post digests,
the viewer-private feature check had zero mismatches, and all 36 paired games
terminated within the declared 32-Command cap. The development arena remains
non-promotional and produced a flat 0.5 payoff matrix.

Semantic transferred perfectly on the Fire Nation Cadets identity holdout
(3/3 seeds), while identity-only transferred on 0/3. That does not establish a
structure effect: structure-shuffled also reached 3/3. On the South Pole
Voyager known-operation composition, semantic reached exact agreement on only
1/3 seeds, identity-only on 2/3, and structure-shuffled on 0/3. The registered
semantic 90% aggregate-competency outcome was refuted: semantic exact agreement
over the 12 evaluation/holdout/frontier rows was 8/12, 10/12, and 8/12 by
seed. Mean semantic p95 latency was 1.263 ms, 1.383x identity-only's 0.913 ms,
so the 2x latency outcome bound passed.

| Arm | Identity holdout exact | Composition exact | Paired strength |
|---|---:|---:|---:|
| semantic | 1.000 | 0.333 | 0.500 |
| identity-only | 0.000 | 0.667 | 0.500 |
| structure-shuffled | 1.000 | 0.000 | 0.500 |

The prompt competencies also show the prototype's narrowness. Semantic seeds
were perfect on priority-offer selection, 0.333 on targeting, and
0.714/1.000/0.714 on combat. These are useful action-level measurements, not a
general gameplay claim.

## Integrity and systems evidence

- Data: 12 train rows; 6 ordinary evaluation; 2 identity holdout; 2
  composition holdout; 2 frontiers. Maximum authoritative choice counts were
  35 target candidates and 64 represented attacker branches.
- Checkpoints: nine verified 31,970-parameter models, 36 epochs each, with
  exact model/data/content/seed manifests. Training used 75.56 wall seconds
  and 18.22 process CPU seconds in aggregate; peak RSS was 399,917,056 bytes.
- Serving: single-decision p50 was 0.336-0.384 ms and p95 was 0.507-1.929 ms.
  True ragged batches of 12 measured 241-3,948 decisions/s. Replay execution
  measured 488.5 authoritative Commands/s.
- Arena: `development_paired_arena_v1`, Otter-Penguin terminal combat fixture,
  36/36 terminal paired-seat games, zero nonterminal receipts, zero rating or
  promotion authority.
- Manifest: digest
  `9d8d2ce00f75d0fb6902cefc778c20683de9ee9a3658e779759f0d1bdd8d40f1`.

## Interpretation

INT-11 closes the Project's missing execution loop: a learned manabot now acts
through the landed INT-2 runtime join rather than a static kata or parallel
legality surface. It does not show that intact program order/hierarchy caused
transfer. The equal identity-holdout behavior of semantic and
structure-shuffled arms suggests that operation content or the public runtime
join is sufficient for this tiny oracle; the composition result is too
seed-sensitive to distinguish that explanation from optimization noise.

The strongest confound is the deliberately small deterministic oracle and
terminal micro-matchup. The matchup proves complete legal execution but is not
sensitive enough to rank the arms. Any follow-up belongs to the Semantic Policy
parent Project and should change the working prototype or workload, not add a
new static proof ladder.
