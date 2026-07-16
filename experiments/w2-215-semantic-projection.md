# W2-215 semantic learning projection baseline

Measured 2026-07-15 on the checked-in UR Lessons versus GW Allies matchup.
The machine-readable receipt is
[`data/w2-215-semantic-projection.json`](data/w2-215-semantic-projection.json).
This is a baseline, not a training result or a performance threshold.

## Provenance

- Measurement code revision:
  `aef56441c03537fe043783d9447ce2fb22a1800c`
- CPython: 3.12.12; native extension:
  `_managym.cpython-312-darwin.so`
- Platform: macOS 26.0.1, arm64, 16 logical CPUs
- Seed: 215; viewer observations: 4,096 across 19 games
- Learning schema hash:
  `21afc6287b081332f1f060ca0cc85f075d2fab5e3397c6d89f55bc941aeceb1f`
- Semantic IR hash:
  `c8bfe15eab35e5953c7a55ba69d83753fa8878b86f9f74e371ec66a233606337`
- ContentPack hash:
  `2444fc15283068d79f74d49f8297bdbe3cc581a62a1eee976b35a8cecd8a0ef4`
- Semantic pack hash:
  `40edd512095e974d73be81b53707090fd3a2e38cefbb9e83b9abcce42d84e39e`

Reproduce with:

```bash
uv run scripts/benchmark_semantic_projection.py --seed 215 --states 4096 \
  --batch-sizes 1,32,256 \
  --revision aef56441c03537fe043783d9447ce2fb22a1800c \
  --measured-at 2026-07-15T19:08:20-07:00 \
  --out experiments/data/w2-215-semantic-projection.json
```

## Encoding size

The immutable catalog contains 31 definitions, 37 programs, 2,088 typed
tokens, and four definition-reference edges. It occupies 13,116 bytes ragged
or 27,384 bytes after deterministic catalog padding. Per visible object,
semantic content is 53 tokens at p50, 95 at p95, and 148 at maximum.

## Latency and throughput

Cold schema/IR validation, tokenization, and exact ContentPack binding took
3.149 ms. Hot viewer projection latency was 9.875 microseconds at p50 and
15.416 microseconds at p95; the maximum observed sample was 403.708
microseconds.

| Batch size | Batch-only observations/s | Encode + batch observations/s | Encode + batch tokens/s |
|---:|---:|---:|---:|
| 1 | 68,842 | 36,783 | 89,974,557 |
| 32 | 360,076 | 59,612 | 145,818,197 |
| 256 | 364,055 | 55,698 | 136,243,587 |

The largest measured 256-observation ragged object batch occupied 253,602
bytes and its padded form occupied 334,080 bytes. Python traced peak allocation
was 1,003,313 bytes. Process RSS increased by 688,128 bytes during the isolated
memory pass; process peak RSS was 269,451,264 bytes.

## Correctness boundary

All 176,802 visible object bindings were admitted. The benchmark recorded zero
projection failures, zero unadmitted visible objects, and zero valid opaque
identity features in `semantic_only` mode. Focused tests separately prove
deterministic encoding and masking, exact environment ContentPack binding,
hidden-state determinization safety, structural recombination, definition-edge
integrity, artifact-header rejection, and unknown schema/opcode rejection.

This task does not interpret semantic IR in the rules engine, alter the fixed
observation/action ABI, train a semantic encoder or decoder, or claim held-out
transfer.
