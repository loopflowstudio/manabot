"""Step 3 design arithmetic: the (hours, dollars) planner over existing evidence.

A sketch for the design table only; the planner software is specified in the protocol.
Every input is a measured figure from steps 1-2, MTG-183 or 07c490, or a labeled
assumption. Run: uv run python scratch/planner_sketch.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------- measured inputs

# Seconds per update per seed, w64 control, 256 streams, one producer.
# A4000 (EPYC 7702): step 2 M2 35.0 (u2-20) / 40.2 (u2-8); S22 50.9; M1 60.2 (u2-20).
# L4: step 1, 25.4 (EPYC 9254) / 37.0 (EPYC 7702) solo, 31.6/41.4 at K=2, 61-63 at K=4.
SECONDS: dict[str, dict[int, float]] = {
    "a4000": {1: 37.0, 2: 50.0, 4: 62.0},
    "l4": {1: 31.0, 2: 36.5, 4: 62.0},
}
USD_PER_HOUR = {"a4000": 0.256, "l4": 0.59}  # billed rates, disk included
CPU32_USD_PER_HOUR = 1.00  # cpu32 on-demand ceiling
GAMES_PER_CPU32_HOUR = 13_000  # 07c490 pass 2: 1,200 pairings x 4 legs in 22 min
EVAL_PASS_SETUP_HOURS = 0.25

# Continuation: 12-hour job windows. MTG-183 windows lost ~20 min to setup and
# CUDA validation; step 2 setup was 1.5 min, resume adds staging. Take 10 min, plus
# step 2's hang rate (1 of 21 attempts) at a 600 s stall watcher and a re-setup.
WINDOW_HOURS = 12.0
WINDOW_LOSS_HOURS = 10 / 60
HANG_RATE = 1 / 21
HANG_LOSS_HOURS = (600 + 600) / 3600
TRAIN_FRACTION = (WINDOW_HOURS - WINDOW_LOSS_HOURS - HANG_RATE * HANG_LOSS_HOURS) / WINDOW_HOURS

# MTG-183 seed 18301 (w13, one seed): rating vs heuristic=1000 on full grids.
CURVE = [(100, 961), (200, 1031), (400, 1077), (800, 1118), (1600, 1166), (3200, 1206), (5950, 1248)]


def elo(updates: float, late_slope: float) -> float:
    """Piecewise log-linear through MTG-183; `late_slope` Elo per doubling past 5,950."""
    if updates <= CURVE[0][0]:
        return CURVE[0][1] - 100 * math.log2(CURVE[0][0] / max(updates, 1))
    for (u0, e0), (u1, e1) in zip(CURVE, CURVE[1:]):
        if updates <= u1:
            return e0 + (e1 - e0) * math.log2(updates / u0) / math.log2(u1 / u0)
    return CURVE[-1][1] + late_slope * math.log2(updates / CURVE[-1][0])


def expected_max(n: int) -> float:
    """E[max of n standard normals] (Blom approximation, exact enough here)."""
    if n <= 1:
        return 0.0
    table = {2: 0.564, 3: 0.846, 4: 1.029, 6: 1.267, 8: 1.424, 12: 1.629, 16: 1.766}
    return table.get(n, 1.0)


# ---------------------------------------------------------------- plans


@dataclass
class Plan:
    name: str
    machine: str
    machines: int
    seeds: int
    final_updates: float
    train_usd: float
    eval_usd: float

    @property
    def usd(self) -> float:
        return self.train_usd + self.eval_usd

    def value(self, late_slope: float, sigma: float, rho: float, sigma_eval: float) -> float:
        # Selection gain: true spread shrunk by measurement noise, discounted by the
        # early-to-final rank correlation for plans that cut before the end.
        shrink = sigma**2 / math.sqrt(sigma**2 + sigma_eval**2) if sigma > 0 else 0.0
        cut = rho if self.name.startswith("halve") else 1.0
        return elo(self.final_updates, late_slope) + expected_max(self.seeds) * shrink * cut


def eval_usd(candidates: int, passes: int, games: int = 2000) -> float:
    if candidates <= 1:
        return 0.0
    hours = candidates * games / GAMES_PER_CPU32_HOUR + EVAL_PASS_SETUP_HOURS
    return passes * hours * CPU32_USD_PER_HOUR


def rate(machine: str, k: int) -> float:
    return 3600 / SECONDS[machine][k] * TRAIN_FRACTION


def plans(hours: float, usd: float, machine: str, max_machines: int) -> list[Plan]:
    price = USD_PER_HOUR[machine]
    out: list[Plan] = []
    for m in range(1, max_machines + 1):
        # Solo: m seeds, each alone on its machine, best of m at the end.
        e = eval_usd(m, 1)
        h = min(hours, (usd - e) / (m * price))
        if h > 1:
            out.append(Plan(f"solo x{m}", machine, m, m, rate(machine, 1) * h, m * h * price, e))
        # Packed: 4m seeds, four per machine throughout, best of 4m at the end.
        e = eval_usd(4 * m, 1)
        h = min(hours, (usd - e) / (m * price))
        if h > 1:
            out.append(Plan(f"pack4 x{m}", machine, m, 4 * m, rate(machine, 4) * h, m * h * price, e))
        # Halving 4m -> 2m -> m per... on m machines: phases 3:4:7 of the window.
        # Phase 1 K=4 on m machines; phase 2 K=2 on m machines; phase 3 solo, the
        # m survivors' best is chosen at the end (m=1: one survivor). Dollar-bound
        # variants release machines instead (not modeled: same machine-hours).
        e = eval_usd(4 * m, 1) + eval_usd(2 * m, 1) + eval_usd(m, 1)
        h = min(hours, (usd - e) / (m * price))
        if h > 1:
            u = h * (3 / 14 * rate(machine, 4) + 4 / 14 * rate(machine, 2) + 7 / 14 * rate(machine, 1))
            out.append(Plan(f"halve x{m}", machine, m, 4 * m, u, m * h * price, e))
    return out


def main() -> None:
    cells = [(h, d) for d in (10, 50, 100) for h in (6, 24, 72)] + [(120, 100), (336, 100)]
    scenarios = {
        "central": dict(late_slope=44.0, sigma=10.0, rho=0.8, sigma_eval=10.0),
        "no-spread": dict(late_slope=44.0, sigma=0.0, rho=0.8, sigma_eval=10.0),
        "wide-flat": dict(late_slope=20.0, sigma=18.0, rho=0.8, sigma_eval=10.0),
    }
    print(f"train fraction per window {TRAIN_FRACTION:.3f}")
    for machine, cap in (("a4000", 3), ("l4", 8)):
        print(f"\n== {machine} (at most {cap} machines)")
        for hours, usd in cells:
            options = plans(hours, usd, machine, cap)
            best = {k: max(options, key=lambda p: p.value(**v)) for k, v in scenarios.items()}
            c = best["central"]
            flip = len({b.name for b in best.values()}) > 1
            line = (
                f"${usd:>3} {hours:>4}h  central: {c.name:<10} seeds {c.seeds:>2} final u{c.final_updates:>7,.0f}"
                f"  ${c.train_usd:6.2f}+{c.eval_usd:4.2f}  elo {c.value(**scenarios['central']):6.0f}"
            )
            if flip:
                line += "  FLIP: " + ", ".join(f"{k}={b.name}" for k, b in best.items() if b.name != c.name)
            print(line)
        # Worked example detail.
    print("\n== ($100, 336 h), one A4000, the three fixed shapes")
    for p in plans(336, 100, "a4000", 1):
        vals = {k: round(p.value(**v)) for k, v in scenarios.items()}
        print(f"{p.name:<10} seeds {p.seeds} u{p.final_updates:,.0f} ${p.usd:.2f} {vals}")


if __name__ == "__main__":
    main()


def break_even(hours: float, usd: float, machine: str, late_slope: float, rho: float) -> float | None:
    """Smallest seed spread (Elo) at which halving x1 beats one solo seed."""
    options = {p.name: p for p in plans(hours, usd, machine, 1)}
    solo, halve = options["solo x1"], options["halve x1"]
    for tenth in range(0, 600):
        sigma = tenth / 10
        if halve.value(late_slope, sigma, rho, 10.0) > solo.value(late_slope, sigma, rho, 10.0):
            return sigma
    return None


if __name__ == "__main__":
    for machine in ("a4000", "l4"):
        for cell in ((336, 100), (120, 100), (72, 50), (72, 100), (24, 10)):
            print(machine, cell, {s: {r: break_even(*cell, machine, s, r) for r in (1.0, 0.8, 0.5)} for s in (44.0, 20.0)})
