"""Supervised fitting of the production memory encoder on a separate hand target.

Only summaries and the public baseline vector feed the linear probe. There is
no action/value loss, recurrent carry or privileged input. Losses average cards
then decisions; uncertainty resamples games, averaging the three fitting seeds.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import sys
import sysconfig
import time
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.env.memory_store import PrefixObservation
from manabot.infra.config import ModelConfig
from manabot.infra.identity import file_sha256
from manabot.infra.source import source_commit
from manabot.policy.prepare import Prepared
from manabot.verify.decision_memory_probe import (
    Corpus,
    bootstrap_mean,
    choose_k,
    collect_games,
    game_losses,
    new_corpus,
    probe_space,
    save_corpus,
)
import managym

Encoder = Literal["none", "transformer", "gru"]
FIT_SEEDS = (39101, 39102, 39103)
MIN_GAMES = 60
MAX_WALL_SECONDS = 7200.0


class HandProbe(nn.Module):
    def __init__(self, corpus: Corpus, encoder: Encoder, summaries: int) -> None:
        super().__init__()
        self.policy = (
            None
            if encoder == "none"
            else ActorCritic(
                probe_space(),
                ModelConfig(
                    world=managym.WORLD_SETUP,
                    hidden_dim=64,
                    decision_memory=encoder,
                    decision_memory_summaries=summaries,
                    attention_on=True,
                    attention_layers=2,
                    policy_film=True,
                    value_aggregation="masked_mean",
                ),
            )
        )
        self.head = nn.Linear(
            len(corpus.rows[0].public) + (0 if self.policy is None else summaries * 64),
            len(corpus.names),
        )
        if self.policy is not None:
            # Everything the public summary method reads is trainable, and no
            # unused board/action/value parameters participate in the optimizer.
            for name, parameter in self.policy.named_parameters():
                parameter.requires_grad_(
                    name.startswith(
                        (
                            "decision_memory.",
                            "semantic_cards.",
                            "name_candidates.embedding.",
                        )
                    )
                )

    def forward(
        self, public: torch.Tensor, prepared: Prepared[torch.Tensor] | None
    ) -> torch.Tensor:
        if self.policy is None:
            return self.head(public)
        assert prepared is not None
        summaries = self.policy.encode_decision_memory(prepared).flatten(1)
        return self.head(torch.cat((public, summaries), -1))


@dataclass(frozen=True)
class FitResult:
    name: str
    encoder: Encoder
    summaries: int
    seed: int
    seconds: float
    epochs: int
    validation: np.ndarray
    weights: Path


def _prepared(
    model: HandProbe, corpus: Corpus, indexes: np.ndarray
) -> Prepared[torch.Tensor] | None:
    if model.policy is None:
        return None
    rows = [corpus.rows[int(i)].observation for i in indexes]
    batch = PrefixObservation(
        {key: np.concatenate([row[key] for row in rows], 0) for key in rows[0]},
        [row.prefixes[0] for row in rows],
        rows[0].memory_capacities,
    ).materialize()
    return on_device(model.policy.prepare(batch), "cpu")


def _predict(
    model: HandProbe,
    corpus: Corpus,
    indexes: np.ndarray,
    *,
    batch_size: int = 16,
    deadline: float = float("inf"),
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indexes), batch_size):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "probe wall cap exhausted during readout; partial evidence retained"
                )
            selected = indexes[start : start + batch_size]
            public = torch.from_numpy(
                np.stack([corpus.rows[int(i)].public for i in selected])
            )
            predictions.append(
                model(public, _prepared(model, corpus, selected)).sigmoid().numpy()
            )
    return np.concatenate(predictions)


def fit(
    corpus: Corpus,
    encoder: Encoder,
    summaries: int,
    seed: int,
    out: Path,
    *,
    epochs: int = 8,
    deadline: float = float("inf"),
) -> FitResult:
    """Select the epoch on validation only; never inspect the test labels."""
    started = time.monotonic()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HandProbe(corpus, encoder, summaries)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.003
    )
    train, validation = corpus.indexes("fit"), corpus.indexes("validation")
    public = torch.from_numpy(np.stack([row.public for row in corpus.rows]))
    validation_labels = np.stack([corpus.labels[int(i)] for i in validation])
    name = "state-only" if encoder == "none" else f"{encoder}-k{summaries}"
    weights = out / f"{name}-{seed}.pt"
    best = float("inf")
    best_predictions: np.ndarray | None = None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), 16):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "probe wall cap exhausted during fitting; partial evidence retained"
                )
            selected = order[start : start + 16]
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(public[selected], _prepared(model, corpus, selected)),
                torch.from_numpy(np.stack([corpus.labels[int(i)] for i in selected])),
            )
            loss.backward()
            optimizer.step()
        predictions = _predict(model, corpus, validation, deadline=deadline)
        # Match the game-weighted readout used for choosing K.
        losses = game_losses(validation_labels, predictions)
        score = float(_by_game(corpus, validation, losses).mean())
        if score < best:
            best, best_predictions = score, predictions
            torch.save(model.state_dict(), weights)
        print(
            f"{name} seed={seed} epoch={epoch + 1}/{epochs} validation={score:.6f}",
            flush=True,
        )
    assert best_predictions is not None
    return FitResult(
        name,
        encoder,
        summaries,
        seed,
        time.monotonic() - started,
        epochs,
        best_predictions,
        weights,
    )


def _by_game(corpus: Corpus, indexes: np.ndarray, losses: np.ndarray) -> np.ndarray:
    games = np.array([corpus.rows[int(i)].game for i in indexes])
    return np.array([losses[games == game].mean() for game in np.unique(games)])


def run_probe(
    out: Path, *, epochs: int = 8, max_games: int = 120, wall_seconds: float = 7200
) -> dict[str, object]:
    """Run a bounded local probe; persist data and weights before its readout.

    A 20-game smoke plus one transformer epoch sizes all fits conservatively.
    An insufficient budget fails explicitly instead of silently omitting arms.
    """
    if epochs < 1 or max_games < MIN_GAMES or not 0 < wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError(
            f"probe needs epochs >=1, max_games >={MIN_GAMES}, and "
            f"0<wall_seconds<={MAX_WALL_SECONDS:g}"
        )
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    sources = (
        "experiments/opponent_memory.py",
        "manabot/backends/torch/decision_memory_probe.py",
        "manabot/verify/decision_memory_probe.py",
        "manabot/backends/torch/policy.py",
        "manabot/backends/torch/_decision_memory.py",
    )
    for name in sources:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    (out / "runtime.json").write_text(
        json.dumps(
            {
                "revision": source_commit(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "native_sha256": file_sha256(
                    root
                    / "managym"
                    / ("_managym" + sysconfig.get_config_var("EXT_SUFFIX"))
                ),
                "sources": {name: file_sha256(root / name) for name in sources},
                "threads": 1,
                "precision": "float32",
            },
            indent=2,
        )
        + "\n"
    )
    torch.set_num_threads(1)
    started = time.monotonic()
    deadline = started + wall_seconds
    corpus = new_corpus()
    collection_seconds = collect_games(corpus, 20, out, deadline=deadline)
    save_corpus(corpus, out)
    # Calibration must contain fit/validation games independently of the final
    # test readout. The deterministic seed schedule is fixed across candidates.
    if not len(corpus.indexes("fit")) or not len(corpus.indexes("validation")):
        raise ValueError("20-game smoke lacks fit/validation rows")
    smoke = fit(
        corpus, "transformer", 4, FIT_SEEDS[0], out, epochs=1, deadline=deadline
    )
    per_game = collection_seconds / 20 + smoke.seconds / 20 * epochs * 15
    games = min(max_games, int(wall_seconds * 0.65 / max(per_game, 0.001)))
    if games < 60:
        raise RuntimeError(
            f"20-game smoke admits only {games} games inside the wall cap; need at least 60"
        )
    (out / "calibration.json").write_text(
        json.dumps(
            {
                "smoke_games": 20,
                "collection_seconds": collection_seconds,
                "fit_epoch_seconds": smoke.seconds,
                "epochs": epochs,
                "games": games,
                "max_games": max_games,
                "wall_cap_seconds": wall_seconds,
            },
            indent=2,
        )
        + "\n"
    )
    collect_games(corpus, games - 20, out, start=20, deadline=deadline)
    for split in ("fit", "validation", "test"):
        selected = corpus.indexes(split)
        for disclosure in ("open", "closed"):
            if (
                len(
                    {
                        corpus.rows[int(i)].game
                        for i in selected
                        if corpus.rows[int(i)].disclosure == disclosure
                    }
                )
                < 2
            ):
                raise ValueError(f"need at least two {disclosure} games in {split}")
    save_corpus(corpus, out)
    candidates: tuple[tuple[Encoder, int], ...] = (
        ("none", 1),
        ("transformer", 1),
        ("transformer", 4),
        ("gru", 1),
        ("gru", 4),
    )
    fits: list[FitResult] = []
    for encoder, k in candidates:
        for seed in FIT_SEEDS:
            if time.monotonic() - started > wall_seconds:
                raise RuntimeError(
                    "probe exhausted wall cap; partial evidence retained, no K selected"
                )
            fits.append(
                fit(corpus, encoder, k, seed, out, epochs=epochs, deadline=deadline)
            )
            print(
                f"fit {encoder} K={k} seed={seed} {fits[-1].seconds:.1f}s", flush=True
            )
    validation = corpus.indexes("validation")
    target = np.stack([corpus.labels[int(i)] for i in validation])
    differences = []
    for encoder in ("transformer", "gru"):
        by_k = {
            k: np.mean(
                [
                    game_losses(target, f.validation)
                    for f in fits
                    if f.encoder == encoder and f.summaries == k
                ],
                axis=0,
            )
            for k in (1, 4)
        }
        differences.append(_by_game(corpus, validation, by_k[4] - by_k[1]))
    costs = {
        k: sum(f.seconds for f in fits if f.encoder != "none" and f.summaries == k)
        for k in (1, 4)
    }
    selected_k, reason = choose_k(np.mean(differences, axis=0), costs)
    # The test split is evaluated only after the K decision is frozen.
    (out / "selection.json").write_text(
        json.dumps({"k": selected_k, "reason": reason}, indent=2) + "\n"
    )
    test = corpus.indexes("test")
    targets = np.stack([corpus.labels[int(i)] for i in test])
    losses: dict[str, list[np.ndarray]] = {
        "analytic": [
            game_losses(targets, np.stack([corpus.rows[int(i)].analytic for i in test]))
        ]
    }
    for result in fits:
        model = HandProbe(corpus, result.encoder, result.summaries)
        model.load_state_dict(torch.load(result.weights, weights_only=True))
        predictions = _predict(model, corpus, test, deadline=deadline)
        np.savez_compressed(
            out / f"predictions-{result.name}-{result.seed}.npz",
            indexes=test,
            probabilities=predictions,
        )
        losses.setdefault(result.name, []).append(game_losses(targets, predictions))
    averaged = {name: np.mean(values, axis=0) for name, values in losses.items()}
    table: dict[str, dict[str, object]] = {}
    for name, row_losses in averaged.items():
        table[name] = {}
        for disclosure in ("open", "closed"):
            mask = np.array(
                [corpus.rows[int(i)].disclosure == disclosure for i in test]
            )
            score = bootstrap_mean(_by_game(corpus, test[mask], row_losses[mask]))
            diff = bootstrap_mean(
                _by_game(
                    corpus, test[mask], (row_losses - averaged["state-only"])[mask]
                )
            )
            table[name][disclosure] = {
                "loss": score,
                "minus_state": diff,
                "games": len({corpus.rows[int(i)].game for i in test[mask]}),
            }
    metadata = [
        {**asdict(f), "validation": f.validation.tolist(), "weights": f.weights.name}
        for f in fits
    ]
    summary: dict[str, object] = {
        "selected_k": selected_k,
        "reason": reason,
        "games": corpus.games,
        "rows": len(corpus.rows),
        "overflows": corpus.overflows,
        "capped": corpus.capped,
        "input_shapes": probe_space().shapes,
        "fit_seeds": FIT_SEEDS,
        "epochs": epochs,
        "smoke": {
            "games": 20,
            "collection_seconds": collection_seconds,
            "fit_epoch_seconds": smoke.seconds,
        },
        "wall_seconds": time.monotonic() - started,
        "cost_usd": 0,
        "table": table,
        "fits": metadata,
        "limits": "Scripted play; coarse public-state baseline excludes known-hand and stack facts; frequency baseline approximates presence; game bootstrap conditional on these fitting seeds; not RL or strength evidence.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("candidate                    open loss                  closed loss")
    for name in table:
        values = []
        for disclosure in ("open", "closed"):
            item = table[name][disclosure]
            assert isinstance(item, dict)
            mean, lo, hi = item["loss"]
            text = f"{mean:.5f} [{lo:.5f},{hi:.5f}]"
            if name not in ("analytic", "state-only"):
                mean, lo, hi = item["minus_state"]
                text += f" diff {mean:+.5f} [{lo:+.5f},{hi:+.5f}]"
            values.append(text)
        print(f"{name:20} {values[0]}   {values[1]}")
    print(
        f"selected K: {selected_k} ({reason})\ngames {corpus.games} rows {len(corpus.rows)} capacity overflows {corpus.overflows} capped {corpus.capped}"
    )
    return summary
