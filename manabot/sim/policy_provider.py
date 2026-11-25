"""Manabot's baseline adapter. Discovery and training metadata stay here.

A factory pins the checkpoint plus runner/backend identity. Independent sessions
share no RNG or model cache. Optional Academy replay/advice is a separate entry
point; this module imports only the engine's public policy contract.
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from urllib.error import URLError
from urllib.request import Request, urlopen

import managym
from managym.policy import (
    JSONValue,
    Policy,
    PolicyDecision,
    PolicyDescriptor,
    PolicyDetail,
    PolicyEvaluation,
    PolicySession,
    PolicySpec,
    PolicyView,
    ValueEstimate,
)

if TYPE_CHECKING:
    from manabot.backends.torch.policy import ActorCritic, PolicyInference
    from manabot.env.env import Env
    from manabot.env.observation import ObservationSpace
    from manabot.model.catalog import CheckpointSummary

# Scripted reference players: no artifact, selected by `config["kind"]`.
_ANCHORS = ("heuristic", "greedy")
DEFAULT_HOSTED_INDEX_URL = "https://policies.manabot.gg/index.json"
_HOSTED_INDEX_TTL_SECONDS = 300
_LOGGER = logging.getLogger(__name__)


def runner_identity() -> str:
    """Bind installed implementation bytes, including encoders and backends."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return f"manabot/{version('manabot')}:{digest.hexdigest()}:torch/{version('torch')}"


def _deck_name(deck: managym.Deck) -> str:
    # Names are presentation only. Admission below always compares full decks,
    # including sideboards; an unrecognized list keeps its own content identity.
    for preset in ("two_deck", "optional_payments", "jeong_increment"):
        for key in managym.authored_deck_keys(preset):
            player = managym.authored_deck_setup(preset, key)
            if deck == managym.Deck(dict(player.decklist), dict(player.sideboard)):
                return key
    return f"custom deck {deck.identity[:10]}"


@dataclass(frozen=True)
class BaselineFactory:
    spec: PolicySpec
    summary: CheckpointSummary | None = None

    def _inspect(self) -> CheckpointSummary | None:
        """Current checkpoint facts, or None for a scripted reference player."""
        from manabot.backends import resolve
        from manabot.model.catalog import inspect_checkpoint

        if self.spec.artifact is None:
            if self.spec.config.get("kind") not in _ANCHORS:
                raise ValueError("Manabot policy requires an artifact.")
            return None
        return inspect_checkpoint(Path(self.spec.artifact), backend=resolve("torch"))

    def describe(self) -> PolicyDescriptor:
        summary = self._inspect()
        if summary is None:
            kind = str(self.spec.config["kind"])
            return PolicyDescriptor(
                kind.title(),
                "manabot",
                runner_identity(),
                config=self.spec.config,
                alias=kind,
            )
        # Discovery's summary carries the run facts; the bytes are re-read here.
        displayed = self.summary or summary
        details = tuple(
            PolicyDetail(label, "not recorded" if value is None else str(value))
            for label, value in (
                ("Run", displayed.run),
                ("Run ID", displayed.run_id),
                ("Stage", displayed.stage),
                ("Weights", displayed.weights),
                ("Updates", displayed.updates),
                ("Transitions", displayed.transitions),
                ("Model", displayed.model),
                ("Parameters", displayed.parameters),
                ("Step", displayed.step),
                ("Source", displayed.source),
            )
        )
        if summary.setups:
            details += (
                PolicyDetail(
                    "Bound decks",
                    " vs ".join(_deck_name(deck) for deck in summary.setups),
                ),
            )

        # Search admission belongs to the shared composition, not a bare flag.
        return PolicyDescriptor(
            displayed.label,
            "manabot",
            runner_identity(),
            {"actor": hashlib.sha256(summary.path.read_bytes()).hexdigest()},
            self.spec.config,
            group=displayed.run or displayed.run_id,
            details=details,
            incompatibility=summary.incompatibility,
            supports_search=summary.supports_search,
            search_reason=summary.search_reason,
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        from manabot.infra.config import ObservationConfig

        summary = self._inspect()
        if summary is None:
            return
        ObservationConfig.from_checkpoint(
            {"policy_input_version": summary.policy_input_version}
        )
        if summary.incompatibility:
            raise ValueError(
                f"{summary.label} cannot play here: {summary.incompatibility}"
            )
        trained = [
            json.dumps({"deck": deck.main, "sideboard": deck.sideboard}, sort_keys=True)
            for deck in summary.setups
        ]
        actual = [
            json.dumps(
                {"deck": dict(config.decklist), "sideboard": dict(config.sideboard)},
                sort_keys=True,
            )
            for config in setup
        ]
        if trained and not (
            sorted(trained) == sorted(actual)
            or (
                len(trained) > 2
                and len(actual) == 2
                and all(deck in trained for deck in actual)
            )
        ):
            bound = " vs ".join(_deck_name(deck) for deck in summary.setups)
            raise ValueError(
                f"Checkpoint setup differs: {summary.label} was trained on {bound}; choose those decks "
                "or another policy."
            )

    def load(self, *, seat: int, seed: int) -> PolicySession:
        if self.spec.artifact is None:
            return AnchorSession(str(self.spec.config["kind"]))
        return BaselineSession(self.spec, seed=seed)


class AnchorSession:
    """A scripted reference player: one legal action, no probabilities or value."""

    def __init__(self, kind: str) -> None:
        from manabot.sim.anchors import ScriptedGreedyPlayer
        from manabot.sim.heuristic import HeuristicPlayer

        self.anchor = (
            HeuristicPlayer() if kind == "heuristic" else ScriptedGreedyPlayer()
        )

    def observe(self, view: PolicyView) -> None:
        pass

    def choose(self, view: PolicyView) -> PolicyDecision:
        # The scripted players read only the raw observation from their Env.
        action = self.anchor.act(
            cast("Env", SimpleNamespace(last_raw_obs=view.observation)), {}
        )
        return PolicyDecision(view.choices[action], view.binding)

    def close(self) -> None:
        pass


class BaselineSession:
    """A loaded checkpoint; search-capable through evaluate and fork."""

    def __init__(self, spec: PolicySpec, *, seed: int) -> None:
        from manabot.backends.torch.checkpoint import load_checkpoint

        assert spec.artifact is not None
        self.spec = spec
        self._rng = random.Random(seed)
        self.agent: ActorCritic | None
        self.space: ObservationSpace | None
        self.agent, self.space = load_checkpoint(spec.artifact)
        # One inference scope per loaded policy caches the card-program table;
        # forks share it and only this loading session closes it.
        self._scope: PolicyInference | None = self.agent.inference_scope()
        self._owns_scope = True

    def observe(self, view: PolicyView) -> None:
        pass

    def choose(self, view: PolicyView) -> PolicyDecision:
        evaluation = self.evaluate(view)
        probabilities = evaluation.probabilities
        action = (
            max(range(len(view.choices)), key=probabilities.__getitem__)
            if self.spec.config.get("deterministic", False)
            else self._rng.choices(
                range(len(view.choices)), weights=probabilities, k=1
            )[0]
        )
        return PolicyDecision(
            view.choices[action], view.binding, probabilities, evaluation.value
        )

    def evaluate(self, view: PolicyView) -> PolicyEvaluation:
        from manabot.backends.torch.bot import evaluate_observation

        if self.agent is None or self.space is None or self._scope is None:
            raise ValueError("Policy session is closed.")
        probabilities, value = evaluate_observation(
            self._scope, self.space, view.observation
        )
        semantic = self.agent.value_semantic or "undeclared critic units"
        if not math.isfinite(value):
            raise ValueError("Policy returned a nonfinite value estimate.")
        if semantic == "signed_outcome":
            # A scalar head regressed on signed outcomes can overshoot its
            # targets; an expected outcome lies in [-1, 1], so project the
            # estimate there. W/D/L expectations are already inside it.
            value = min(1.0, max(-1.0, value))
        return PolicyEvaluation(
            probabilities, ValueEstimate(value, view.seat, semantic)
        )

    def fork(self, *, seed: int) -> BaselineSession:
        # Model parameters are read-only; history-free evaluation mutates no model state.
        branch = copy.copy(self)
        branch._rng = random.Random(seed)
        branch._owns_scope = False
        return branch

    def close(self) -> None:
        if self._owns_scope and self._scope is not None:
            self._scope.__exit__(None, None, None)
        self._scope = None
        self.agent = None
        self.space = None


@dataclass(frozen=True)
class _UnavailablePolicy:
    descriptor: PolicyDescriptor

    def describe(self) -> PolicyDescriptor:
        return self.descriptor

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        raise ValueError(self.descriptor.incompatibility)

    def load(self, *, seat: int, seed: int) -> PolicySession:
        raise ValueError(self.descriptor.incompatibility)


def _ssl_context() -> Any:
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _hosted_cache_root() -> Path:
    base = os.environ.get("MANABOT_CACHE_DIR")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".cache" / "manabot"


def _hosted_enabled() -> bool:
    flag = os.environ.get("ACADEMY_HOSTED_POLICIES")
    if flag is not None:
        return flag.strip().lower() not in ("0", "false", "no", "off")
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


def _fetch_hosted_index() -> dict[str, Any] | None:
    if not _hosted_enabled():
        return None
    url = os.environ.get("ACADEMY_POLICIES_URL", DEFAULT_HOSTED_INDEX_URL)
    cache_dir = _hosted_cache_root() / "hosted"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_cache = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.json"
    if (
        index_cache.is_file()
        and (time.time() - index_cache.stat().st_mtime) < _HOSTED_INDEX_TTL_SECONDS
    ):
        try:
            loaded = json.loads(index_cache.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except (OSError, ValueError):
            pass
    try:
        req = Request(url, headers={"User-Agent": "manabot-academy"})
        with urlopen(req, timeout=5, context=_ssl_context()) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            index_cache.write_bytes(raw)
            return data
    except (OSError, URLError, ValueError) as exc:
        _LOGGER.debug("Hosted policy index unavailable (%s): %s", url, exc)
        if index_cache.is_file():
            try:
                loaded = json.loads(index_cache.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded
            except (OSError, ValueError):
                pass
    return None


def _ensure_hosted_artifact(url: str, sha256: str, size_bytes: int | None) -> Path:
    cache_dir = _hosted_cache_root() / "sha256"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".safetensors" if url.endswith(".safetensors") else ".pt"
    dest = cache_dir / f"{sha256}{suffix}"
    if dest.is_file():
        if hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
            return dest
    alt = cache_dir / sha256
    if alt.is_file() and hashlib.sha256(alt.read_bytes()).hexdigest() == sha256:
        return alt
    req = Request(url, headers={"User-Agent": "manabot-academy"})
    with urlopen(req, timeout=60, context=_ssl_context()) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ValueError(
            f"Hosted checkpoint digest mismatch for {url}: expected {sha256[:12]}, got {digest[:12]}."
        )
    if size_bytes is not None and len(data) != size_bytes:
        raise ValueError(
            f"Hosted checkpoint size mismatch for {url}: expected {size_bytes}, got {len(data)}."
        )
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)
    return dest


@dataclass(frozen=True)
class HostedPolicy:
    """Remote content-addressed checkpoint served from ``policies.manabot.gg``."""

    entry: dict[str, Any]
    is_default: bool = False

    def _cached_path(self) -> Path | None:
        sha = str(self.entry["sha256"])
        cache_dir = _hosted_cache_root() / "sha256"
        for candidate in (
            cache_dir / f"{sha}.safetensors",
            cache_dir / f"{sha}.pt",
            cache_dir / sha,
        ):
            if candidate.is_file():
                return candidate
        return None

    def _resolve_local(self) -> Path:
        return _ensure_hosted_artifact(
            str(self.entry["url"]),
            str(self.entry["sha256"]),
            int(self.entry["size_bytes"])
            if self.entry.get("size_bytes") is not None
            else None,
        )

    def describe(self) -> PolicyDescriptor:
        sha = str(self.entry["sha256"])
        name = str(self.entry.get("name") or self.entry["id"])
        details = [
            PolicyDetail("Source", "policies.manabot.gg"),
            PolicyDetail("ID", str(self.entry["id"])),
        ]
        if self.entry.get("experiment"):
            details.append(PolicyDetail("Experiment", str(self.entry["experiment"])))
        if self.entry.get("run"):
            details.append(PolicyDetail("Run", str(self.entry["run"])))
        if self.entry.get("updates") is not None:
            details.append(PolicyDetail("Updates", str(self.entry["updates"])))
        cached = self._cached_path()
        incompatibility: str | None = None
        supports_search = True
        search_reason: str | None = None
        if cached is not None:
            from manabot.backends import resolve
            from manabot.model.catalog import inspect_checkpoint

            summary = inspect_checkpoint(cached, backend=resolve("torch"))
            incompatibility = summary.incompatibility
            supports_search = summary.supports_search
            search_reason = summary.search_reason
        raw_settings = self.entry.get("trained_settings")
        trained_settings = (
            dict(raw_settings) if isinstance(raw_settings, dict) else None
        )
        return PolicyDescriptor(
            name=name,
            provider="manabot",
            runner=runner_identity(),
            artifacts={"actor": sha},
            config={"hosted_id": str(self.entry["id"])},
            group="Hosted default" if self.is_default else "Hosted",
            details=tuple(details),
            alias="manabot" if self.is_default else str(self.entry["id"]),
            incompatibility=incompatibility,
            supports_search=supports_search,
            search_reason=search_reason,
            recommended=self.is_default,
            summary=str(self.entry["description"])
            if self.entry.get("description")
            else None,
            trained_settings=trained_settings,
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        local = self._resolve_local()
        BaselineFactory(PolicySpec("manabot", str(local))).admit(setup)

    def load(self, *, seat: int, seed: int) -> PolicySession:
        local = self._resolve_local()
        return BaselineSession(PolicySpec("manabot", str(local)), seed=seed)


class BaselineProvider:
    def policy(self, spec: PolicySpec) -> BaselineFactory:
        if set(spec.config) - {"kind", "deterministic"}:
            raise ValueError("Unknown manabot policy config fields.")
        return BaselineFactory(spec)

    def _hosted_policies(self) -> list[Policy]:
        index = _fetch_hosted_index()
        if not index:
            return []
        default_id = index.get("default_policy")
        out: list[Policy] = []
        for entry in index.get("policies") or ():
            if not isinstance(entry, dict):
                continue
            if entry.get("provider", "manabot") != "manabot":
                continue
            if not entry.get("id") or not entry.get("sha256") or not entry.get("url"):
                continue
            out.append(HostedPolicy(entry=entry, is_default=entry["id"] == default_id))
        return out

    def discover(self, config: dict[str, JSONValue] | None) -> list[Policy]:
        from manabot.backends import resolve
        from manabot.model.catalog import discover_checkpoints

        ambient = config is None
        if config is None:
            explicit_paths = [
                p
                for p in os.environ.get("ACADEMY_PLAY_CHECKPOINTS", "").split(
                    os.pathsep
                )
                if p
            ]
            explicit_runs_dir = os.environ.get("MANABOT_RUNS_DIR") or None
            paths: list[str] = explicit_paths
            root: str | None = explicit_runs_dir
        else:
            raw_paths = config.get("paths", [])
            raw_root = config.get("runs_dir")
            if (
                not isinstance(raw_paths, list)
                or not all(isinstance(p, str) for p in raw_paths)
                or (raw_root is not None and not isinstance(raw_root, str))
            ):
                raise ValueError(
                    "manabot discovery requires paths and an optional runs_dir string."
                )
            paths = list(raw_paths)
            root = raw_root
        if paths or root is not None:
            summaries, missing = discover_checkpoints(
                [Path(p).expanduser() for p in paths],
                backend=resolve("torch"),
                runs_dir=Path(root).expanduser() if root is not None else None,
            )
        else:
            summaries, missing = [], []
        hosted = self._hosted_policies() if ambient else []
        hosted_shas = {
            str(p.entry["sha256"]) for p in hosted if isinstance(p, HostedPolicy)
        }
        return [
            *hosted,
            *(
                self.policy(PolicySpec("manabot", config={"kind": kind}))
                for kind in _ANCHORS
                if ambient
            ),
            *(
                BaselineFactory(PolicySpec("manabot", str(summary.path)), summary)
                for summary in summaries
                if summary.sha256 not in hosted_shas
            ),
            *(
                _UnavailablePolicy(
                    PolicyDescriptor(
                        entry.label,
                        "manabot",
                        runner_identity(),
                        {"actor": entry.sha256},
                        details=(PolicyDetail("Source", "tournament"),),
                        incompatibility=entry.reason,
                    )
                )
                for entry in missing
            ),
        ]


provider = BaselineProvider()
