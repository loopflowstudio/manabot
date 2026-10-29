"""Own the campaign's loopback vLLM process and retain its startup identity.

vLLM lives in a separately locked Linux environment. The ordinary game process
keeps its existing Torch/Transformers versions and exchanges exact token IDs.
The caller must close the server even when game startup or collection fails.
"""

from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import TracebackType
from urllib.error import URLError
from urllib.request import urlopen

from huggingface_hub import snapshot_download

from manabot.llm.cache import MODEL_PATTERNS
from manabot.llm.schema import InferenceConfig, RuntimeIdentity
from manabot.llm.tokenization import load_tokenizer
from manabot.llm.vllm_runtime import SERVICE_PORT, SERVICE_URL, verify_service


def server_command(
    config: InferenceConfig,
    model: str,
    concurrency: int,
    speculative_tokens: int | None = None,
) -> list[str]:
    """No remote template, prefix reuse, quantization or generation defaults.

    `speculative_tokens` enables the checkpoint's own multi-token-prediction
    head as a draft; rejection sampling keeps the target distribution, but the
    server argv, and so the behavior version, records it. This is the base argv;
    the derived behavior route is appended after identity hashing.
    """
    speculative = (
        []
        if speculative_tokens is None
        else [
            "--speculative-config",
            json.dumps(
                {"method": "mtp", "num_speculative_tokens": speculative_tokens},
                separators=(",", ":"),
            ),
        ]
    )
    return [
        "uv",
        "run",
        "--no-project",
        "--no-sync",
        "--python",
        # Resolving this symlink selects system Python and loses the venv.
        str(Path(".venv-vllm/bin/python").absolute()),
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVICE_PORT),
        "--dtype",
        config.precision,
        "--max-model-len",
        str(config.context_tokens),
        "--max-num-seqs",
        str(concurrency),
        "--gpu-memory-utilization",
        "0.85",
        "--language-model-only",
        "--skip-tokenizer-init",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--max-num-batched-tokens",
        "4096",
        *speculative,
    ]


class VLLMServer(AbstractContextManager["VLLMServer"]):
    def __init__(
        self,
        config: InferenceConfig,
        out: Path,
        concurrency: int,
        speculative_tokens: int | None = None,
    ) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.out = out
        self.log = (out / "vllm.log").open("wb")
        self.config = config
        self.concurrency = concurrency
        self.speculative_tokens = speculative_tokens
        self._stop = threading.Event()

    def __enter__(self) -> "VLLMServer":
        started = time.monotonic()
        try:
            model = snapshot_download(
                self.config.model_id,
                revision=self.config.model_revision,
                allow_patterns=MODEL_PATTERNS,
                local_files_only=True,
            )
            tokenizer = load_tokenizer(self.config, offline=True)
            command = server_command(
                self.config, model, self.concurrency, self.speculative_tokens
            )
            # Query the separate server interpreter, not the client's packages.
            probe = subprocess.run(
                command[: command.index("-m")]
                + [
                    "-c",
                    "import json,platform,torch; from importlib.metadata import version; print(json.dumps(dict(runtime_version=version('vllm'),torch_version=torch.__version__,python_version=platform.python_version(),device_name=torch.cuda.get_device_name())))",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.log.write(probe.stdout.encode() + probe.stderr.encode())
            self.log.flush()
            probe.check_returncode()
            facts = json.loads(probe.stdout)
            identity = RuntimeIdentity(
                runtime="vllm",
                **facts,
                chat_template=tokenizer.get_chat_template(),
                generation_config={
                    # The derived route cannot participate in its own hash.
                    "server_argv_without_identity": command,
                    "concurrency": self.concurrency,
                    "client": "exact-token-completions-v1",
                    "model_revision": self.config.model_revision,
                    "tokenizer_revision": self.config.tokenizer_revision,
                    "precision": self.config.precision,
                    "context_tokens": self.config.context_tokens,
                },
            )
            version = identity.behavior_version(self.config)
            command = [*command, "--served-model-name", version]
            (self.out / "vllm-command.json").write_text(json.dumps(command, indent=2))
            self.process = subprocess.Popen(
                command,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(
                "vLLM server starting; startup details retained in vllm.log", flush=True
            )
            deadline = started + 900
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM exited during startup: {self.process.returncode}; see vllm.log"
                    )
                try:
                    with urlopen(f"{SERVICE_URL}/health", timeout=2) as response:
                        if response.status == 200:
                            verify_service(version, timeout=2)
                            (self.out / "vllm-identity.json").write_text(
                                identity.model_dump_json(indent=2)
                            )
                            self.seconds = time.monotonic() - started
                            print(
                                f"vLLM server healthy after {self.seconds:.3f}s",
                                flush=True,
                            )
                            threading.Thread(
                                target=self._relay_metrics, daemon=True
                            ).start()
                            return self
                except (URLError, TimeoutError):
                    pass
                time.sleep(1)
            raise TimeoutError("vLLM startup exceeded 900 seconds")
        except BaseException:
            self.close()
            raise

    def _relay_metrics(self, interval: float = 60) -> None:
        """Copy vLLM's periodic throughput and speculative-acceptance lines to stdout.

        Job stdout is retrievable while the campaign runs; vllm.log arrives only
        with the outputs. At most one line of each kind per interval.
        """
        last: dict[str, float] = {}
        with (self.out / "vllm.log").open("r", errors="replace") as log:
            while not self._stop.is_set():
                line = log.readline()
                if not line:
                    self._stop.wait(1)
                    continue
                for kind in ("throughput", "SpecDecoding"):
                    now = time.monotonic()
                    if kind in line and now - last.get(kind, -interval) >= interval:
                        last[kind] = now
                        print(f"VLLM {line.strip()}", flush=True)

    def close(self) -> None:
        self._stop.set()
        if self.process is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
            self.process.wait(timeout=5)
        self.log.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
