# SPS: Parallel Stepping — Validation

## Verify (Rust)

```bash
cargo test
cargo test --features python
cargo clippy --all-targets --all-features -- -D warnings
```

## Verify (Python)

```bash
pip install -e managym && pytest tests/env/ tests/verify/ -v
python -m manabot.verify.step0_env_sanity
```

## Measure

**Scaling sweep:**
```bash
for n in 1 4 16 64 128; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
done
```

**Single-thread baseline (isolates parallelism contribution):**
```bash
RAYON_NUM_THREADS=1 python scripts/bench_ab.py --a rust --b rust --num-envs 16 --rounds 5
```

"Better" = near-linear SPS scaling up to core count. Target: >100k env-only SPS at 16 envs on M4 Max.
