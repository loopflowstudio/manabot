# W2-214 preliminary typed-opcode alignment probe

Status: **PASS**. Decision: **preliminary_opcode_alignment_signal**.

Run with:

```sh
PYTHONHASHSEED=0 uv run scripts/run_opcode_alignment.py \
  --workload experiments/workloads/opcode-alignment-v1.json \
  --out experiments/data/opcode-alignment-v1.json \
  --report experiments/opcode-alignment-v1.md
```

## Claim boundary

This probe measures typed-opcode alignment only. It does not establish semantic compositional transfer, AST understanding, gameplay strength, PPO transfer, search strength, or win rate.
This result does **not** close Rules Semantic KR6 and is not evidence
that the model reads abilities as a language. The supervised label is an
opcode directly present in the input; the encoder is order-invariant; the
production policy/action ABI is unchanged.

## Four-arm result

| Arm | input | head | zero-shot heldout cluster accuracy (95% Wilson CI) | exact-program retrain |
|---|---|---|---:|---:|
| `card_id_legacy` | opaque CardDefId | legacy fixed | 0.0% [0.0%, 24.2%] | 100.0% [75.8%, 100.0%] |
| `card_id_structured` | opaque CardDefId | structured | 0.0% [0.0%, 24.2%] | 100.0% [75.8%, 100.0%] |
| `semantic_card_id_structured` | program + CardDefId | structured | 16.7% [4.7%, 44.8%] | 100.0% [75.8%, 100.0%] |
| `semantic_only_structured` | program only | structured | 91.7% [64.6%, 98.5%] | 100.0% [75.8%, 100.0%] |

The independent unit is one held-out program × model seed (4 × 3 = 12).
Seat and candidate-order repeats are deterministic sensitivity checks and
are excluded from uncertainty. The same programs, permutations, seeds, and
optimizer budget are used across arms. Opaque identity slots are hash-ordered,
not IR rows, and held-out identity embeddings receive no zero-shot updates.

## Paired cluster contrasts

| Contrast (left − right) | Δ accuracy (paired bootstrap CI) | wins/losses/ties | exact sign p |
|---|---:|---:|---:|
| decoder: CardDefId structured − legacy | 0.0% [0.0%, 0.0%] | 0/0/12 | 1 |
| semantic input: semantic-only − CardDefId structured | 91.7% [75.0%, 100.0%] | 11/0/1 | 0.0009766 |
| semantic+identity − CardDefId structured | 16.7% [0.0%, 41.7%] | 2/0/10 | 0.5 |
| identity ablation: semantic-only − semantic+identity | 75.0% [50.0%, 100.0%] | 9/0/3 | 0.003906 |
| opcode present − opcode masked | 8.3% [0.0%, 25.0%] | 1/0/11 | 1 |
| ordered − token shuffled | 0.0% [0.0%, 0.0%] | 0/0/12 | 1 |

The structured decoder alone provides no alignment benefit over the legacy
head. Semantic-only input aligns strongly with the visible opcode. Adding an
untrained identity channel causes severe interference; removing it is the
largest paired improvement. This is a useful architecture warning.

## Semantic controls and holdout validity

| Control | zero-shot cluster accuracy (95% Wilson CI) |
|---|---:|
| opcode tokens masked | 83.3% [55.2%, 95.3%] |
| program tokens shuffled | 91.7% [64.6%, 98.5%] |

The shuffled-token control is expected to match because the current encoder
uses masked mean/max pooling. It cannot distinguish same-token programs with
different AST order. The opcode-masked control measures how much alignment
comes from other correlated tokens.

Only 2/4 held-out programs have a normalized AST shape absent from training, and 2/4 have full symbolic primitive closure. Per-program duplicate/gap rows are in the
JSON receipt. These failures force the opcode-alignment claim downgrade.

## Binder-local contract smoke tests

- Program tokens: p50 29, p95 64, max 75 against the observed 75-token pack maximum; 0 local overflows/truncations.
- ContentPack reorder + checkpoint rebind: 100.0% exact.
- Token-kind/opcode numeric-ID permutation: 100.0% exact.
- Unknown opcodes and changed enum-domain schemas fail closed.

These are self-consistency checks of this binder and generated pack, not an
independent semantic oracle. Likewise, the 64-branch >32 frontier is inherited
W2-189 engine/adapter evidence and is unrelated to classifier training.

## Performance

Latency, throughput, parameter bytes, and process peak RSS are recorded
for every arm/seed in the JSON receipt. Latency is batch-1. RSS is the
shared-process high-water mark, not an isolated model-memory comparison.
Limited retraining exposes each exact held-out program and therefore measures
exact-program memorization, not few-shot semantic transfer.

## What remains

- Build an order-sensitive semantic encoder and evaluate same-token, different-AST programs.
- Pre-register a larger holdout with normalized-AST novelty and full symbolic primitive closure.
- Join the semantic encoder and structured decoder to PPO/all admitted decision families.
- Add enum-domain identity to the projection before claiming arbitrary schema-table rebinding.
- Run held-out gameplay and win-rate transfer after that join.

The enum follow-on is structural: v1 emits one flat `enum` token without
its source domain. This experiment can safely rebind token-kind and opcode
IDs, but arbitrary enum-table permutation would be ambiguous and is
therefore rejected rather than counted as robustness.
