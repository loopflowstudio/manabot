"""Local fine-tuned Qwen adapter: native text in, one legal reference out.

This action-only provider never invents a probability vector or value. Model,
config and tokenizer files are pinned together; each seat owns its runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

from manabot.sim.policy_provider import runner_identity
import managym
from managym.policy import PolicyDecision, PolicyDescriptor, PolicySpec, PolicyView


@dataclass(frozen=True)
class QwenFactory:
    spec: PolicySpec

    def describe(self) -> PolicyDescriptor:
        if self.spec.artifact is None:
            raise ValueError("Qwen requires a local saved model directory.")
        root = Path(self.spec.artifact).expanduser().resolve()
        config = json.loads((root / "config.json").read_text())
        if not str(config.get("model_type", "")).startswith("qwen"):
            raise ValueError("Qwen provider requires a Qwen model configuration.")
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix
            in (".safetensors", ".json", ".jinja", ".txt", ".model", ".tiktoken")
        )
        if not any(p.suffix == ".safetensors" for p in files):
            raise ValueError("Qwen directory has no safetensors weights.")
        artifacts = {}
        for path in files:
            with path.open("rb") as stream:
                artifacts[str(path.relative_to(root))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        return PolicyDescriptor(
            root.name,
            "qwen",
            f"{runner_identity()}:transformers/{version('transformers')}",
            artifacts,
            self.spec.config,
            search_reason="Qwen v1 returns actions without full legal probabilities or a value evaluator.",
            supports_search=False,
        )

    def admit(self, setup: Sequence[managym.PlayerConfig]) -> None:
        # Native setup admission belongs to the engine. This adapter consumes
        # its complete text authority rather than a trained tensor layout.
        del setup

    def load(self, *, seat: int, seed: int) -> QwenSession:
        del seat, seed  # Greedy decoding has no mutable RNG.
        return QwenSession(self.spec)


class QwenSession:
    def __init__(self, spec: PolicySpec) -> None:
        self.spec = spec
        self.runtime = LocalQwenRuntime(
            str(Path(spec.artifact).expanduser()),
            device=str(spec.config.get("device", "cpu")),
            precision=str(spec.config.get("precision", "float32")),
        )

    def observe(self, view: PolicyView) -> None:
        pass

    def choose(self, view: PolicyView) -> PolicyDecision:
        from manabot.semantic.text_observation import render_decision

        if view.capture_json is None:
            raise ValueError("Qwen requires a viewer-safe native decision capture.")
        decision = render_decision(view.capture_json)
        if len(decision.actions) != len(view.choices):
            raise ValueError("Text choices differ from native legal actions.")
        prompt = (
            decision.text
            + "\nReturn exactly one listed action reference, with no other text."
        )
        answer = self.runtime.choose(
            prompt,
            max_new_tokens=int(self.spec.config.get("max_new_tokens", 32)),
            seconds=float(self.spec.config.get("seconds", 30)),
        )
        for index, action in enumerate(decision.actions):
            if answer == action.reference:
                return PolicyDecision(view.choices[index], view.binding)
        raise ValueError(
            "Qwen did not return a legal action reference; decision remains pending."
        )

    def close(self) -> None:
        self.runtime = None


class QwenProvider:
    def policy(self, spec: PolicySpec) -> QwenFactory:
        if set(spec.config) - {"device", "precision", "max_new_tokens", "seconds"}:
            raise ValueError("Unknown Qwen policy config fields.")
        if spec.config.get("device", "cpu") not in ("cpu", "cuda") or spec.config.get(
            "precision", "float32"
        ) not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported Qwen device or precision.")
        for key, default in (("max_new_tokens", 32), ("seconds", 30)):
            value = spec.config.get(key, default)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
                or (key == "max_new_tokens" and type(value) is not int)
            ):
                raise ValueError(f"Qwen {key} must be positive.")
        return QwenFactory(spec)


provider = QwenProvider()


class LocalQwenRuntime:
    """Fine-tuned local Qwen safetensors, with bounded greedy generation.

    The provider verifies the entire saved model/tokenizer tree before loading.
    No Hub resolution, remote code or automatic model substitution is allowed.
    """

    def __init__(self, path: str, *, device: str, precision: str) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        config = AutoConfig.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        if not str(config.model_type).startswith("qwen"):
            raise ValueError("Qwen provider requires a Qwen model configuration.")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=getattr(torch, precision),
            )
            .to(device)
            .eval()
        )

    def choose(self, prompt: str, *, max_new_tokens: int, seconds: float) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit is not None and inputs.input_ids.shape[1] + max_new_tokens > limit:
            raise ValueError("The complete decision exceeds the Qwen context window.")
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                max_time=seconds,
            )
        if time.monotonic() - started > seconds:
            raise TimeoutError(
                "Qwen decision deadline exhausted; decision remains pending."
            )
        return self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
