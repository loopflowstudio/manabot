"""Exact token boundaries and failure accounting without GPU allocation."""

from concurrent.futures import Future
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import TracebackType
from typing import Callable
from urllib.request import Request

from etude.launch.records import Job
from huggingface_hub import _snapshot_download, snapshot_download
from huggingface_hub._tree_cache import TreeCacheEntry
from huggingface_hub.errors import IncompleteSnapshotError
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from experiments import qwen_mtg183_vllm as campaign
from manabot.llm import tokenization, vllm_runtime, vllm_server
from manabot.llm.schema import InferenceConfig, Message, RuntimeIdentity, Sampling
from manabot.verify.llm_game import GameSummary


def test_server_uses_its_own_virtual_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An actual uv-created venv must survive interpreter symlinks and a parent venv."""
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(tmp_path / ".venv-vllm")],
        check=True,
        capture_output=True,
        timeout=20,
    )
    config = InferenceConfig.model_validate_json(
        campaign.original.INFERENCE.read_text()
    )
    monkeypatch.setenv("VIRTUAL_ENV", sys.prefix)
    monkeypatch.chdir(tmp_path)
    command = vllm_server.server_command(config, "fixture", 4)
    probe = subprocess.run(
        command[: command.index("-m")] + ["-c", "import sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert Path(probe.stdout.strip()) == tmp_path / ".venv-vllm"


def test_speculative_draft_is_part_of_the_declared_server() -> None:
    config = campaign.TREATMENTS["budget"].load()
    plain = vllm_server.server_command(config, "fixture", 32)
    drafted = vllm_server.server_command(config, "fixture", 32, 1)
    assert "--speculative-config" not in plain
    assert drafted[len(plain) :] == [
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":1}',
    ]


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
    for owner in (tokenization, vllm_runtime):
        monkeypatch.setattr(
            owner, "snapshot_download", lambda *args, **kwargs: str(tmp_path)
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
    monkeypatch.setattr(
        vllm_runtime, "urlopen", _serving(identity.behavior_version(config))
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


def _serving(served: str) -> Callable[[str, float], Reply]:
    """A model list naming one served behavior route."""

    def models(url: str, timeout: float) -> Reply:
        assert url.endswith("/v1/models")
        return Reply({"data": [{"id": served}]})

    return models


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "model",
        "missing-model",
        "prompt",
        "terminator",
        "limit",
        "transport",
        "late",
    ],
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
        assert body["model"] == runtime.identity.behavior_version(runtime.config)
        if failure == "transport":
            raise TimeoutError("fixture transport deadline")
        if failure == "late":
            monkeypatch.setattr(vllm_runtime.time, "monotonic", lambda: deadline)
        reply = Reply(
            {
                "model": "other-version" if failure == "model" else body["model"],
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
                ],
            }
        )
        if failure == "missing-model":
            del reply.payload["model"]
        return reply

    monkeypatch.setattr(vllm_runtime, "urlopen", request)
    deadline = time.monotonic() + 10
    if failure:
        with pytest.raises(
            TimeoutError if failure in ("transport", "late") else ValueError
        ):
            runtime.generate(prompt, deadline)
    else:
        response = runtime.generate(prompt, deadline)
        assert response.text == "thought </think> A0"
        assert response.token_ids == (4, 2, 3, 1)
        assert response.reasoning_tokens == response.answer_tokens == 1
        assert response.finish_reason == "eos"


def test_sampled_requests_send_the_declared_treatment_and_prompt_seed(
    runtime: vllm_runtime.VLLMRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling = Sampling(
        temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, seed=183
    )
    config = runtime.config.model_copy(
        update={"decoding": "sampled", "sampling": sampling}
    )
    assert runtime.identity.behavior_version(
        config
    ) != runtime.identity.behavior_version(runtime.config)
    monkeypatch.setattr(
        vllm_runtime, "urlopen", _serving(runtime.identity.behavior_version(config))
    )
    sampled = vllm_runtime.VLLMRuntime(config, identity_path=tmp_path / "identity.json")
    bodies: list[dict[str, object]] = []

    def request(req: Request, timeout: float) -> Reply:
        body = json.loads(req.data or b"{}")
        bodies.append(body)
        choice = {
            "index": 0,
            "finish_reason": "stop",
            "prompt_token_ids": body["prompt"],
            "token_ids": [4, 2, 3, 1],
        }
        return Reply({"model": body["model"], "choices": [choice]})

    monkeypatch.setattr(vllm_runtime, "urlopen", request)
    first = sampled.encode((Message(role="user", content="A0"),))
    second = sampled.encode((Message(role="user", content="thought A0"),))
    for prompt in (first, first, second):
        sampled.generate(prompt, time.monotonic() + 10)
    body = bodies[0]
    assert (body["temperature"], body["top_p"], body["top_k"]) == (1.0, 0.95, 20)
    assert (body["min_p"], body["presence_penalty"]) == (0, 1.5)
    assert body["seed"] == bodies[1]["seed"] != bodies[2]["seed"]
    assert body["seed"] == sampling.request_seed(first.token_ids)


@pytest.mark.parametrize(
    ("first", "second", "forced"),
    [
        # Unfinished reasoning at the budget: insert the close, then answer.
        ([4, 4, 4], [3, 1], True),
        # Closed within the budget but the answer was cut: continue unchanged.
        ([4, 2, 3], [1], False),
        # Finished within the budget: one request.
        ([2, 3, 1], None, False),
    ],
)
def test_budget_forcing_closes_unfinished_reasoning(
    runtime: vllm_runtime.VLLMRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first: list[int],
    second: list[int] | None,
    forced: bool,
) -> None:
    config = runtime.config.model_copy(update={"thinking_budget": 3})
    monkeypatch.setattr(
        vllm_runtime, "urlopen", _serving(runtime.identity.behavior_version(config))
    )
    budgeted = vllm_runtime.VLLMRuntime(
        config, identity_path=tmp_path / "identity.json"
    )
    replies = [first, second]
    bodies: list[dict[str, object]] = []

    def request(req: Request, timeout: float) -> Reply:
        body = json.loads(req.data or b"{}")
        bodies.append(body)
        tokens = replies[len(bodies) - 1]
        choice = {
            "index": 0,
            "finish_reason": "stop" if tokens and tokens[-1] == 1 else "length",
            "prompt_token_ids": body["prompt"],
            "token_ids": tokens,
        }
        return Reply({"model": body["model"], "choices": [choice]})

    monkeypatch.setattr(vllm_runtime, "urlopen", request)
    prompt = budgeted.encode((Message(role="user", content="A0"),))
    response = budgeted.generate(prompt, time.monotonic() + 10)
    assert len(bodies) == (1 if second is None else 2)
    assert bodies[0]["max_tokens"] == 3
    inserted = (2,) if forced else ()
    head = tuple(first) + inserted
    if second is not None:
        assert bodies[1]["prompt"] == [*prompt.token_ids, *head]
        assert bodies[1]["max_tokens"] == 8 - len(head)
    assert response.token_ids == head + tuple(second or ())
    assert response.text.endswith("A0") and response.finish_reason == "eos"
    assert response.forced_close_at == (3 if forced else None)
    assert response.inserted_token_ids == inserted
    assert response.answer_tokens == 1


def test_thinking_budget_requires_thinking_and_answer_room() -> None:
    fields = json.loads(campaign.original.INFERENCE.read_text())
    with pytest.raises(ValueError, match="thinking budget"):
        InferenceConfig.model_validate({**fields, "thinking_budget": 2048})
    with pytest.raises(ValueError, match="thinking budget"):
        InferenceConfig.model_validate(
            {**fields, "enable_thinking": None, "thinking_budget": 16}
        )


@pytest.mark.parametrize(
    ("arm", "output_tokens", "context_tokens"),
    [
        ("sampled", 16_384, 65_536),
        ("budget", 4096, 16_384),
        ("budget-64k", 4096, 65_536),
        ("nothink", 512, 16_384),
    ],
)
def test_declared_arm_reaches_the_frozen_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    output_tokens: int,
    context_tokens: int,
) -> None:
    # Replace only native artifact preparation; no job is submitted or rented.
    def native_job(native_artifact: str, native_source: str) -> Job:
        return Job(command="fixture", max_hours=11)

    monkeypatch.setattr(campaign.original, "runs", native_job)
    job = campaign.runs("unused", "unused", arm=arm, games=1)
    assert job.config is not None
    config = campaign.Campaign.model_validate_json(job.config)
    games = campaign._prepare_schedule(config, tmp_path)
    assert len(games) == 1
    effective = InferenceConfig.model_validate_json(
        games[0].config.inference.read_text()
    )
    assert (effective.max_new_tokens, effective.context_tokens) == (
        output_tokens,
        context_tokens,
    )
    source = InferenceConfig.model_validate_json(config.inference.read_text())
    # The larger context changes no other part of the treatment.
    assert effective.model_dump(exclude={"context_tokens"}) == source.model_dump(
        exclude={"context_tokens"}
    )


def test_sampled_decoding_requires_its_settings() -> None:
    fields = json.loads(campaign.original.INFERENCE.read_text())
    with pytest.raises(ValueError, match="sampling settings"):
        InferenceConfig.model_validate({**fields, "decoding": "sampled"})
    with pytest.raises(ValueError, match="sampling settings"):
        InferenceConfig.model_validate(
            {
                **fields,
                "sampling": {"temperature": 1, "top_p": 1, "top_k": 1, "seed": 0},
            }
        )


@pytest.mark.parametrize(
    ("model_eos", "expected"), [(7, {1, 7}), ([7, 8], {1, 7, 8}), (None, {1})]
)
def test_generation_defaults_can_come_from_model_config(
    runtime: vllm_runtime.VLLMRuntime,
    tmp_path: Path,
    model_eos: int | list[int] | None,
    expected: set[int],
) -> None:
    (tmp_path / "generation_config.json").unlink()
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "eos_token_id": model_eos})
    )
    loaded = vllm_runtime.VLLMRuntime(
        runtime.config, identity_path=tmp_path / "identity.json"
    )
    assert loaded.eos_ids == expected


@pytest.mark.parametrize(
    "changed", ["service", "model", "tokenizer", "treatment", "runtime"]
)
def test_service_admission_rejects_changed_identity(
    runtime: vllm_runtime.VLLMRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    config = runtime.config
    identity = runtime.identity
    if changed == "model":
        config = config.model_copy(update={"model_revision": "c" * 40})
    elif changed == "tokenizer":
        config = config.model_copy(update={"tokenizer_revision": "c" * 40})
    elif changed == "treatment":
        config = config.model_copy(update={"max_new_tokens": 9})
    elif changed == "runtime":
        identity = identity.model_copy(update={"runtime_version": "other"})
    else:
        monkeypatch.setattr(vllm_runtime, "urlopen", _serving("wrong-service"))
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(identity.model_dump_json())
    with pytest.raises(ValueError, match="does not acknowledge"):
        vllm_runtime.VLLMRuntime(config, identity_path=identity_path)


@pytest.mark.parametrize("mismatch", [False, True])
def test_server_admits_its_derived_route_and_reaps_on_mismatch(
    runtime: vllm_runtime.VLLMRuntime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: bool,
) -> None:
    launched: list[str] = []
    terminated: list[int] = []
    snapshots: list[tuple[str, str]] = []
    waited = False

    class Process:
        pid = 12345
        returncode: int | None = None

        def __init__(self, command: list[str], **kwargs: object) -> None:
            launched.extend(command)

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            nonlocal waited
            waited = True
            self.returncode = 0
            return 0

    class Health(Reply):
        status = 200

    def snapshot(repo: str, *, revision: str, **kwargs: object) -> str:
        assert kwargs["local_files_only"] is True
        snapshots.append((repo, revision))
        return str(tmp_path)

    def probe(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        facts = runtime.identity.model_dump(
            include={
                "runtime_version",
                "torch_version",
                "python_version",
                "device_name",
            }
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(facts), "")

    def models(url: str, timeout: float) -> Reply:
        version = launched[launched.index("--served-model-name") + 1]
        return Reply({"data": [{"id": "wrong-service" if mismatch else version}]})

    monkeypatch.setattr(vllm_server, "snapshot_download", snapshot)
    monkeypatch.setattr(tokenization, "snapshot_download", snapshot)
    monkeypatch.setattr(vllm_server.subprocess, "run", probe)
    monkeypatch.setattr(vllm_server.subprocess, "Popen", Process)
    monkeypatch.setattr(
        vllm_server.os, "killpg", lambda pid, sig: terminated.append(pid)
    )
    monkeypatch.setattr(vllm_server, "urlopen", lambda *args, **kwargs: Health({}))
    monkeypatch.setattr(vllm_runtime, "urlopen", models)
    server = vllm_server.VLLMServer(runtime.config, tmp_path, 4)
    receipt = tmp_path / "vllm-identity.json"
    if mismatch:
        with pytest.raises(ValueError, match="does not acknowledge"):
            with server:
                pytest.fail("a foreign service must not be admitted")
        assert not receipt.exists()
    else:
        with server:
            identity = RuntimeIdentity.model_validate_json(receipt.read_text())
            version = identity.behavior_version(runtime.config)
            base_argv = identity.generation_config["server_argv_without_identity"]
            assert isinstance(base_argv, list)
            assert launched == [
                *base_argv,
                "--served-model-name",
                version,
            ]
            assert json.loads((tmp_path / "vllm-command.json").read_text()) == launched
            client = vllm_runtime.VLLMRuntime(runtime.config, identity_path=receipt)
            assert client.behavior_version == version
    model = (runtime.config.model_id, runtime.config.model_revision)
    tokenizer = (runtime.config.tokenizer_id, runtime.config.tokenizer_revision)
    # Startup loads model then tokenizer; an admitted client reloads the tokenizer.
    assert snapshots == [model, tokenizer] + ([] if mismatch else [tokenizer])
    assert server.log.closed and waited and terminated


def test_long_reasoning_budget_preserves_deals_and_reaches_server(
    tmp_path: Path,
    runtime: vllm_runtime.VLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = campaign.Campaign()
    games = campaign._prepare_schedule(config, tmp_path)
    original = campaign.original.schedule(config)
    assert len(games) == len(original) == 100
    for game, expected in zip(games, original, strict=True):
        assert game.model_dump(
            exclude={"config": {"inference"}}
        ) == expected.model_dump(exclude={"config": {"inference"}})
        assert game.config.max_seconds == config.seconds == 36000
    effective = InferenceConfig.model_validate_json(
        (tmp_path / "inference.json").read_text()
    )
    command = vllm_server.server_command(effective, "fixture", config.concurrency)
    assert command[command.index("--max-model-len") + 1] == "65536"
    assert command[command.index("--max-num-seqs") + 1] == "8"
    monkeypatch.setattr(
        vllm_runtime, "urlopen", _serving(runtime.identity.behavior_version(effective))
    )
    runtime = vllm_runtime.VLLMRuntime(
        effective, identity_path=tmp_path / "identity.json"
    )

    def request(req: Request, timeout: float) -> Reply:
        body = json.loads(req.data or b"{}")
        assert body["max_tokens"] == 16384
        assert 1700 < timeout <= 1800
        return Reply(
            {
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "prompt_token_ids": body["prompt"],
                        "token_ids": [4, 2, 3, 1],
                    }
                ],
            }
        )

    monkeypatch.setattr(vllm_runtime, "urlopen", request)
    prompt = runtime.encode((Message(role="user", content="A0"),))
    assert (
        runtime.generate(
            prompt, time.monotonic() + effective.deadline_seconds
        ).finish_reason
        == "eos"
    )


@pytest.fixture
def campaign_identity(tmp_path: Path) -> RuntimeIdentity:
    identity = RuntimeIdentity(
        runtime="vllm",
        runtime_version="fixture",
        torch_version="fixture",
        python_version="fixture",
        device_name="fixture",
        chat_template="fixture",
        generation_config={},
    )
    (tmp_path / "vllm-identity.json").write_text(identity.model_dump_json())
    return identity


def _use_inline_pool(monkeypatch: pytest.MonkeyPatch, concurrency: int) -> None:
    """Settle submitted game fixtures immediately without spawning game workers."""

    class Pool:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["max_workers"] == concurrency

        def __enter__(self) -> "Pool":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def submit(
            self,
            fn: Callable[
                [campaign.original.ScheduledGame, Path, float],
                campaign.original.GameResult,
            ],
            game: campaign.original.ScheduledGame,
            out: Path,
            deadline: float,
        ) -> Future[campaign.original.GameResult]:
            future: Future[campaign.original.GameResult] = Future()
            future.set_result(fn(game, out, deadline))
            return future

    monkeypatch.setattr(campaign, "ProcessPoolExecutor", Pool)


@pytest.mark.parametrize("concurrency", [4, 8, 16])
def test_failed_batch_retains_every_admitted_game_and_stops_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    concurrency: int,
    campaign_identity: RuntimeIdentity,
) -> None:
    config = campaign.Campaign(concurrency=concurrency)
    games = campaign._prepare_schedule(config, tmp_path)
    # Once frozen, execution must not depend on the source inference defaults.
    monkeypatch.setattr(campaign.original, "INFERENCE", tmp_path / "absent.json")
    submitted: list[int] = []

    def play(
        game: campaign.original.ScheduledGame, out: Path, deadline: float
    ) -> campaign.original.GameResult:
        assert game == games[len(submitted)]
        submitted.append(game.index)
        admission = json.loads((out / "campaign.json").read_text())
        assert admission["attempted"] == len(submitted)
        assert game.index in admission["pending_games"]
        assert admission["session_closed"] is False
        return campaign.original.GameResult(
            game=game,
            summary=GameSummary(
                mode="fixture", completed=False, failure="injected deadline"
            ),
        )

    _use_inline_pool(monkeypatch, concurrency)
    monkeypatch.setattr(campaign, "play_game", play)
    campaign.worker(config, tmp_path)
    receipt = json.loads((tmp_path / "campaign.json").read_text())
    assert len(submitted) == len(set(submitted)) == concurrency
    assert receipt["planned"] == 100
    assert receipt["attempted"] == receipt["settled"] == concurrency
    assert receipt["completed"] == 0 and receipt["session_closed"] is True
    assert len(json.loads((tmp_path / "results.json").read_text())) == concurrency
    assert json.loads((tmp_path / "games.json").read_text())["mode"] == "concurrent"
    effective = InferenceConfig.model_validate_json(
        (tmp_path / "inference.json").read_text()
    )
    assert effective.deadline_seconds == 1800 and effective.max_new_tokens == 16384
    assert effective.context_tokens == 65536
    assert effective.enable_thinking is True


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ({"model_behavior_version": "other"}, "model behavior other is not cohort"),
        ({"opponent_sha256": "0" * 64}, "is not frozen"),
        ({"model_behavior_version": None}, "missing a frozen player identity"),
        ({"opponent_sha256": None}, "missing a frozen player identity"),
    ],
)
@pytest.mark.parametrize("changed_game", [0, 1])
def test_changed_player_identity_stops_cohort_and_is_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    campaign_identity: RuntimeIdentity,
    changed: dict[str, str | None],
    reason: str,
    changed_game: int,
) -> None:
    submitted: list[int] = []
    config = campaign.Campaign(concurrency=1)
    campaign._prepare_schedule(config, tmp_path)
    inference = InferenceConfig.model_validate_json(
        (tmp_path / "inference.json").read_text()
    )
    expected_version = campaign_identity.behavior_version(inference)

    def play(
        game: campaign.original.ScheduledGame, out: Path, deadline: float
    ) -> campaign.original.GameResult:
        admission = json.loads((out / "campaign.json").read_text())
        assert admission["model_behavior_version"] == expected_version
        summary = GameSummary(
            mode="fixture",
            completed=True,
            model_behavior_version=expected_version,
            opponent_sha256=campaign.original.CHECKPOINT_SHA256,
        )
        if len(submitted) == changed_game:
            summary = summary.model_copy(update=changed)
        submitted.append(game.index)
        return campaign.original.GameResult(game=game, summary=summary)

    _use_inline_pool(monkeypatch, config.concurrency)
    monkeypatch.setattr(campaign, "play_game", play)
    campaign.worker(config, tmp_path)
    receipt = json.loads((tmp_path / "campaign.json").read_text())
    expected_games = changed_game + 1
    assert (
        receipt["attempted"]
        == receipt["settled"]
        == receipt["completed"]
        == expected_games
    )
    assert receipt["model_behavior_version"] == expected_version
    assert reason in receipt["stop_reason"]
    results = json.loads((tmp_path / "results.json").read_text())
    assert len(results) == expected_games
    for field, value in changed.items():
        assert results[-1]["summary"][field] == value


def test_expired_campaign_retains_all_unattempted_games(
    tmp_path: Path, campaign_identity: RuntimeIdentity
) -> None:
    config = campaign.Campaign(seconds=1)
    campaign._prepare_schedule(config, tmp_path)
    campaign.worker(config, tmp_path)
    receipt = json.loads((tmp_path / "campaign.json").read_text())
    assert receipt["attempted"] == receipt["completed"] == 0
    assert receipt["planned"] == 100
    assert receipt["session_closed"] is True
    assert receipt["stop_reason"] == "campaign time reserve reached"


@pytest.mark.parametrize("attempt_usd", ["0", "-1", "9.41", "NaN", "Infinity"])
def test_attempt_beyond_remaining_budget_is_refused_before_launch(
    tmp_path: Path, attempt_usd: str
) -> None:
    # Missing provenance files prove refusal precedes any launch preparation.
    with pytest.raises(ValueError, match="attempt_usd"):
        campaign.runs(
            str(tmp_path / "absent.json"), str(tmp_path / "absent.json"), attempt_usd
        )


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
    for module in (owner, tokenization):
        monkeypatch.setattr(
            module, "snapshot_download", partial(snapshot_download, cache_dir=cache_dir)
        )

    class AdmittedSnapshot(Exception):
        pass

    def tokenizer(path: str, **kwargs: object) -> None:
        assert Path(path) == snapshot
        raise AdmittedSnapshot

    monkeypatch.setattr(tokenization.AutoTokenizer, "from_pretrained", tokenizer)
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
