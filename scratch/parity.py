"""One-time MTG-329 fresh-run proof; reference code stays in a temporary archive.

Run with uv run --no-sync --with <EXP-56 checkout> python scratch/parity.py.
Each training process has a 120-second timeout and uses one CPU thread.
"""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import tarfile
import tempfile

REFERENCE = "6ce618b8adacee3a4894d05572399853680bb9fe"


def train(root: Path, out: Path, method: str, reference: bool) -> None:
    # Import only after selecting the source tree, in a fresh process.
    sys.path.insert(0, str(root))
    from manabot.infra.config import ModelConfig
    from manabot.recipes.builders import gray_ogre_match
    from manabot.training.regime import DampedPolicyLearning, Learning

    learning = (
        Learning(epochs=1, minibatches=1, ema=0.9)
        if method == "ppo"
        else DampedPolicyLearning(gradient="damped_policy")
    )
    model = ModelConfig(hidden_dim=16, num_attention_heads=2)
    if reference:
        from manabot.training.execution import execute_regime
        from manabot.training.regime import Execution, TrainingRegime, TrainSelfPlay
        from manabot.verify.store import VerifyStore
        import managym

        regime = TrainingRegime(
            id="parity",
            world=managym.WORLD_VERSION,
            match=gray_ogre_match(),
            agent=model,
            stages=[
                TrainSelfPlay(
                    id="self-play",
                    operation="train_self_play",
                    behavior="current-self",
                    updates=2,
                    streams=2,
                    transitions=16,
                    learning_seats="both",
                    batch_closure="complete-games",
                    learning=learning,
                    execution=Execution(
                        device="cpu",
                        threads=1,
                        workers=1,
                        bucketed_shapes=False,
                        live_tokens=False,
                        wall_seconds=120,
                    ),
                )
            ],
            wall_seconds=120,
            schedule_clock="iteration_fraction",
        )
        with VerifyStore(out.with_suffix(".sqlite")) as store:
            record = execute_regime(regime, 197, out, store)
        assert record.status == "completed", record.error
    else:
        from manabot.training.loops.self_play import SelfPlay, self_play

        settings = SelfPlay(
            match=gray_ogre_match(),
            model=model,
            learning=learning,
            streams=2,
            transitions=16,
            checkpoint_every=1,
        )
        self_play(settings, out, seed=197, updates=2)


def main() -> None:
    import torch

    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="mtg329-parity-") as temporary:
        work = Path(temporary)
        reference = work / "reference"
        reference.mkdir()
        archive = subprocess.check_output(["git", "archive", REFERENCE], cwd=root)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(reference, filter="data")
        files = {
            str(path.relative_to(reference)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in reference.rglob("*")
            if path.is_file()
        }
        native_name = f"_managym{sysconfig.get_config_var('EXT_SUFFIX')}"
        native = root / "managym" / native_name
        (reference / "managym" / native_name).symlink_to(native)
        files[f"managym/{native_name}"] = hashlib.sha256(
            native.read_bytes()
        ).hexdigest()
        (reference / ".manabot-source.json").write_text(
            json.dumps({"commit": REFERENCE, "files": files})
        )
        print(
            f"reference={REFERENCE} native_sha256={files[f'managym/{native_name}']}",
            flush=True,
        )
        for method in ("ppo", "damped"):
            for label, tree in (("reference", reference), ("current", root)):
                subprocess.run(
                    [
                        str(root / ".venv/bin/python")
                        if label == "reference"
                        else sys.executable,
                        str(script),
                        str(tree),
                        str(work / f"{method}-{label}"),
                        method,
                        label,
                    ],
                    cwd=tree,
                    check=True,
                    timeout=120,
                )
            current = work / f"{method}-current"
            state = torch.load(current / "state.pt", weights_only=False)
            assert state["update"] == 2
            for weights in ("raw", "ema"):
                old = torch.load(
                    work / f"{method}-reference" / f"self-play-{weights}.pt",
                    weights_only=False,
                )["model_state_dict"]
                new = torch.load(
                    current / "checkpoints" / f"00000002-{weights}.pt",
                    weights_only=False,
                )["model_state_dict"]
                assert old.keys() == new.keys() == state[weights].keys()
                differences = [
                    name for name in old if not torch.equal(old[name], new[name])
                ]
                assert not differences, f"{method}/{weights}: {differences}"
                assert all(torch.equal(new[name], state[weights][name]) for name in new)
                print(
                    f"{method}/{weights}: {len(old)} tensors bitwise equal (checkpoint and resume state)",
                    flush=True,
                )
            print(f"{method} counts={state['counts']}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        train(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
            sys.argv[3],
            sys.argv[4] == "reference",
        )
    else:
        main()
