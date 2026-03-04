# Throughput benchmark (Rust backend)

Date: 2026-03-03
Environment: local macOS arm64, Python 3.12 `.venv`, Rust `managym` backend

## Workload
- 2 players (`gaea`, `urza`) using identical 40-card decks
- Deck: Mountain=12, Forest=12, Llanowar Elves=10, Grey Ogre=6
- Policy: always take action index `0`
- 500 games, `skip_trivial=True`, profiler enabled

## Command

```bash
source .venv/bin/activate
python - <<'PY'
import time
import managym

DECK = {"Mountain": 12, "Forest": 12, "Llanowar Elves": 10, "Grey Ogre": 6}

env = managym.Env(seed=7, skip_trivial=True, enable_profiler=True)
p1 = managym.PlayerConfig("gaea", DECK)
p2 = managym.PlayerConfig("urza", DECK)

total_steps = 0
start = time.perf_counter()
for _ in range(500):
    env.reset([p1, p2])
    done = truncated = False
    while not (done or truncated):
        _, _, done, truncated, _ = env.step(0)
        total_steps += 1
elapsed = time.perf_counter() - start

print("games=500")
print(f"steps={total_steps}")
print(f"seconds={elapsed:.4f}")
print(f"games_per_sec={500 / elapsed:.2f}")
print(f"steps_per_sec={total_steps / elapsed:.2f}")
PY
```

## Result
- games: **500**
- steps: **40,500**
- wall time: **1.6816 s**
- throughput: **297.34 games/s**
- throughput: **24,084.82 steps/s**

## C++ parity note
A local C++ baseline could not be re-measured in this sandbox because the CMake
toolchain is unavailable (`cmake: command not found`) and the old C++ build path
has been removed in this cutover.
