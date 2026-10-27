"""Exact token boundaries and failure accounting without GPU allocation."""

from concurrent.futures import Future
from functools import partial
import json
from pathlib import Path
import tarfile
import time
from types import TracebackType
from typing import Callable
from urllib.request import Request

from huggingface_hub import _snapshot_download, snapshot_download
from huggingface_hub._tree_cache import TreeCacheEntry
from huggingface_hub.errors import IncompleteSnapshotError
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from experiments import qwen_mtg183_vllm as campaign
from manabot.llm import vllm_runtime, vllm_server
from manabot.llm.schema import InferenceConfig, Message, RuntimeIdentity
from manabot.verify.llm_game import GameSummary


@pytest.fixture
def runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> vllm_runtime.VLLMRuntime:
    impl = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[EOS]": 1, "</think>": 2, "A0": 3, "thought": 4},
            unk_token="[UNK]",
        )
    )
    impl.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=impl,
        eos_token="[EOS]",
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{{ m['content'] }}{% endfor %}",
    )
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 1}')
    identity = RuntimeIdentity(
        runtime="vllm",
        runtime_version="fixture",
        torch_version="fixture",
        python_version="fixture",
        device_name="fixture",
        chat_template=tokenizer.get_chat_template(),
        generation_config={},
    )
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(identity.model_dump_json())
    monkeypatch.setattr(
        vllm_runtime, "snapshot_download", lambda *args, **kwargs: str(tmp_path)
    )
    config = InferenceConfig(
        model_id="fixture/model",
        model_revision="a" * 40,
        tokenizer_id="fixture/model",
        tokenizer_revision="b" * 40,
        device="cuda",
        precision="bfloat16",
        context_tokens=100,
        max_new_tokens=8,
        deadline_seconds=10,
        fallback_seed=37,
        enable_thinking=True,
    )
    return vllm_runtime.VLLMRuntime(config, identity_path=identity_path)


class Reply:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    def __enter__(self) -> "Reply":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass


@pytest.mark.parametrize(
    "failure", [None, "prompt", "terminator", "limit", "transport"]
)
def test_exact_tokens_and_untrusted_server_reply(
    runtime: vllm_runtime.VLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    prompt = runtime.encode((Message(role="user", content="A0"),))

    def request(req: Request, timeout: float) -> Reply:
        assert 0 < timeout <= 10
        body = json.loads(req.data or b"{}")
        assert body["prompt"] == list(prompt.token_ids)
        assert body["temperature"] == 0 and body["max_tokens"] == 8
        assert body["return_token_ids"] is True and body["add_special_tokens"] is False
        if failure == "transport":
            raise TimeoutError("fixture transport deadline")
        return Reply(
            {
                "choices": [
                    {
                        "index": 0,
                        "text": "untrusted server text is not used",
                        "finish_reason": "stop",
                        "prompt_token_ids": [99]
                        if failure == "prompt"
                        else list(prompt.token_ids),
                        "token_ids": [4, 2, 3]
                        if failure == "terminator"
                        else [4] * 9
                        if failure == "limit"
                        else [4, 2, 3, 1],
                    }
                ]
            }
        )

    monkeypatch.setattr(vllm_runtime, "urlopen", request)
    if failure:
        with pytest.raises(TimeoutError if failure == "transport" else ValueError):
            runtime.generate(prompt, time.monotonic() + 10)
    else:
        response = runtime.generate(prompt, time.monotonic() + 10)
        assert response.text == "thought </think> A0"
        assert response.token_ids == (4, 2, 3, 1)
        assert response.reasoning_tokens == response.answer_tokens == 1
        assert response.finish_reason == "eos"


def test_failed_batch_retains_every_admitted_game_and_stops_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[int] = []

    class Pool:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["max_workers"] == 4

        def __enter__(self) -> "Pool":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def submit(
            self,
            fn: Callable[..., campaign.original.GameResult],
            game: campaign.original.ScheduledGame,
            *args: object,
        ) -> Future[campaign.original.GameResult]:
            submitted.append(game.index)
            admission = json.loads((tmp_path / "campaign.json").read_text())
            assert admission["attempted"] == len(submitted)
            assert game.index in admission["pending_games"]
            assert admission["session_closed"] is False
            future: Future[campaign.original.GameResult] = Future()
            future.set_result(
                campaign.original.GameResult(
                    game=game,
                    summary=GameSummary(
                        mode="fixture", completed=False, failure="injected deadline"
                    ),
                )
            )
            return future

    monkeypatch.setattr(campaign, "ProcessPoolExecutor", Pool)
    campaign.worker(campaign.Campaign(), tmp_path)
    receipt = json.loads((tmp_path / "campaign.json").read_text())
    assert len(submitted) == len(set(submitted)) == 4
    assert receipt["planned"] == 100 and receipt["attempted"] == receipt["settled"] == 4
    assert receipt["completed"] == 0 and receipt["session_closed"] is True
    assert len(json.loads((tmp_path / "results.json").read_text())) == 4
    assert json.loads((tmp_path / "games.json").read_text())["mode"] == "concurrent"


def test_expired_campaign_retains_all_unattempted_games(tmp_path: Path) -> None:
    campaign.worker(campaign.Campaign(seconds=1), tmp_path)
    receipt = json.loads((tmp_path / "campaign.json").read_text())
    assert receipt["attempted"] == receipt["completed"] == 0
    assert receipt["planned"] == 100
    assert receipt["session_closed"] is True
    assert receipt["stop_reason"] == "campaign time reserve reached"


def test_collection_waits_for_game_owner_to_finish(tmp_path: Path) -> None:
    out = tmp_path / "output"
    out.mkdir()
    record = tmp_path / "game-records" / "game-096"
    record.mkdir(parents=True)
    (record / "summary.json").write_text("{}")
    campaign.original.collect(out, settled_games=set())
    assert not list(out.iterdir())
    (record / "actor-startup.json").write_text('{"seconds": 2}')
    campaign.original.collect(out, settled_games={record.name})
    with tarfile.open(out / "game-096.tar.gz") as archive:
        assert "game-096/actor-startup.json" in archive.getnames()


@pytest.mark.parametrize("server", [False, True])
def test_offline_snapshot_matches_prepared_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: vllm_runtime.VLLMRuntime,
    server: bool,
) -> None:
    """Exercise Hub's real completeness check with uncached repository docs."""
    config = runtime.config.model_copy(
        update={"tokenizer_revision": runtime.config.model_revision}
    )
    cache_dir = tmp_path / "hub"
    snapshot = (
        cache_dir / "models--fixture--model" / "snapshots" / config.model_revision
    )
    snapshot.mkdir(parents=True)
    required = ["config.json", "tokenizer.json", "model.safetensors"]
    for filename in required:
        (snapshot / filename).write_text("{}")

    def tree(*args: object) -> dict[str, TreeCacheEntry]:
        return {
            name: TreeCacheEntry(size=2, blob_id="0" * 40)
            for name in [*required, "README.md", "LICENSE", ".gitattributes"]
        }

    monkeypatch.setattr(_snapshot_download, "read_tree_cache", tree)
    owner = vllm_server if server else vllm_runtime
    monkeypatch.setattr(
        owner, "snapshot_download", partial(snapshot_download, cache_dir=cache_dir)
    )

    class AdmittedSnapshot(Exception):
        pass

    def tokenizer(path: str, **kwargs: object) -> None:
        assert Path(path) == snapshot
        raise AdmittedSnapshot

    monkeypatch.setattr(owner.AutoTokenizer, "from_pretrained", tokenizer)
    identity = tmp_path / "identity.json"
    identity.write_text(runtime.identity.model_dump_json())

    def construct() -> None:
        if server:
            with vllm_server.VLLMServer(config, tmp_path, 4):
                pytest.fail("fixture must stop before process startup")
        else:
            vllm_runtime.VLLMRuntime(config, identity_path=identity)

    with pytest.raises(AdmittedSnapshot):
        construct()
    (snapshot / ("model.safetensors" if server else "tokenizer.json")).unlink()
    with pytest.raises(IncompleteSnapshotError):
        construct()
