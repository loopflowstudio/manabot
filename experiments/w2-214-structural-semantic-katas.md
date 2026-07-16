# W2-214 structural semantic katas

Status: **REDESIGN**

Decision: **`REDESIGN optimization_or_capacity_unresolved`**

World: **offline static semantic diagnostic; no gameplay ABI or world change**

## Claim boundary

This experiment may nominate one static encoder candidate for W2-213. It does not establish semantic recombination, dynamic binding, card transfer, executable rules parity, gameplay strength, or integration readiness.

A nomination is only a candidate intake for W2-213. Dynamic binding,
held-out recombination, card transfer, and gameplay integration remain untested.

## Per-kata result

| Kata | bag accuracy | structural accuracy | uplift | structural Brier | structural NLL |
|---|---:|---:|---:|---:|---:|
| `order` | 50.0% [50.0%, 50.0%] | 100.0% [100.0%, 100.0%] | 50.0% | 0.0000 | 0.0027 |
| `hierarchy` | 50.0% [50.0%, 50.0%] | 100.0% [100.0%, 100.0%] | 50.0% | 0.0000 | 0.0026 |
| `field_role` | 50.0% [50.0%, 50.0%] | 55.0% [41.1%, 68.9%] | 5.0% | 0.2340 | 0.6539 |
| `argument_binding` | 50.0% [50.0%, 50.0%] | 80.0% [46.0%, 114.0%] | 30.0% | 0.1003 | 0.2860 |
| `target_choice_role` | 50.0% [50.0%, 50.0%] | 75.6% [44.5%, 106.7%] | 25.6% | 0.1234 | 0.3537 |

Training seed is the independent unit; brackets are two-sided 95% t intervals over five seeds.

## Aggregate result

| Arm | train accuracy | validation accuracy | test accuracy | Brier | NLL | ECE (5 bins) |
|---|---:|---:|---:|---:|---:|---:|
| `bag_v1` | 50.0% | 50.0% | 50.0% | 0.2500 | 0.6932 | 0.0027 |
| `relational_semantic_encoder_v1` | 82.0% | 82.0% | 82.1% | 0.0915 | 0.2598 | 0.0134 |

## Seed receipts

| Arm | seed | selected step | train | validation | test | parameters | model p50/p95 (µs) | projector+model p50/p95 (µs) | batch-128 examples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bag_v1` | 21401 | 540 | 50.0% | 50.0% | 50.0% | 9128 | 40.2/41.8 | 64.8/68.3 | 42811 |
| `bag_v1` | 21402 | 520 | 50.0% | 50.0% | 50.0% | 9128 | 39.3/41.7 | 65.0/66.6 | 46296 |
| `bag_v1` | 21403 | 540 | 50.0% | 50.0% | 50.0% | 9128 | 39.3/39.6 | 66.6/74.0 | 42748 |
| `bag_v1` | 21404 | 540 | 50.0% | 50.0% | 50.0% | 9128 | 40.2/40.8 | 65.8/68.0 | 44147 |
| `bag_v1` | 21405 | 620 | 50.0% | 50.0% | 50.0% | 9128 | 39.3/42.2 | 66.1/69.1 | 45860 |
| `relational_semantic_encoder_v1` | 21401 | 800 | 75.2% | 74.4% | 75.0% | 8838 | 203.5/208.5 | 497.8/508.6 | 1254 |
| `relational_semantic_encoder_v1` | 21402 | 780 | 90.0% | 90.0% | 90.0% | 8838 | 200.5/230.3 | 510.9/643.0 | 1187 |
| `relational_semantic_encoder_v1` | 21403 | 800 | 70.0% | 70.0% | 70.0% | 8838 | 206.5/227.6 | 510.7/581.1 | 1240 |
| `relational_semantic_encoder_v1` | 21404 | 800 | 90.0% | 90.0% | 90.0% | 8838 | 206.5/211.7 | 501.2/578.4 | 1247 |
| `relational_semantic_encoder_v1` | 21405 | 800 | 84.6% | 85.6% | 85.6% | 8838 | 205.6/248.8 | 504.3/515.2 | 1248 |

## Gates and instrument audit

- Bag exact symmetry: **True**.
- Structural trainability: **False**.
- Static semantic accuracy: **False**.
- Calibration: **True**.
- Matched parameter/CPU cost: **False**.
- Aggregate uplift t95 lower bound: 20.8%.
- Maximum parameter difference: 3.2%.
- Worst projector+model p95 ratio: 9.657x.
- Worst projector+model throughput ratio: 0.026x.
- Normalized-program split overlaps: {"train_test": 0, "train_validation": 0, "validation_test": 0}.
- Nuisance split overlaps: {"train_test": 0, "train_validation": 0, "validation_test": 0}.
- Pair-template split overlaps: {"train_test": 0, "train_validation": 0, "validation_test": 0}.
- Intentional family-skeleton overlaps: {"train_test": 5, "train_validation": 5, "validation_test": 5}.
- Definition-reference tokens: 0; opaque identity fields tensorized: [].
- All checked label-independent field contingencies are exactly balanced; full tables are in the JSON receipt.

## Provenance and budget

- Pre-registration revision: `71c9961c0bf401efb7c4009b62aeb43b0f3b326c`.
- Contract SHA-256: `c7afb920522e7d5926982b4dd2d53cdfbf8fc862ad96f2a91cc16c9149d33e8a`.
- Suite SHA-256: `5595ce579017c4ec84b8746cb30a3f4bb09a69e4801ccaf7748467f7bec2f948`.
- Compiler SHA-256: `f10f34d7b2e458a0ea01b124261a0c0fbbda4648e4452f277254aa6cc3250367`.
- Learning-schema SHA-256: `a156c592414d0d4838c5423e2cb471fc49a4450f21d62e5e5c198ba948ae7034`.
- Optimizer steps: 8000 / 8000.
- Presented examples: 512000 / 512000.
- Wall clock: 123.93s / 1800s.

## Result-contingent next step

`REDESIGN optimization_or_capacity_unresolved`

The declared redesign branch is the only authorized continuation; no model scaling or gameplay integration follows from this result.
