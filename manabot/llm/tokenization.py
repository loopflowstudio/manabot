"""Pinned chat encoding and lossless response decoding shared by LLM backends."""

from collections.abc import Callable

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from manabot.llm.cache import TOKENIZER_PATTERNS
from manabot.llm.schema import (
    EncodedPrompt,
    GeneratedResponse,
    InferenceConfig,
    Message,
)


def load_tokenizer(
    config: InferenceConfig, *, offline: bool
) -> PreTrainedTokenizerBase:
    """Resolve the pinned Hub revision, never a same-named local directory."""
    snapshot = snapshot_download(
        config.tokenizer_id,
        revision=config.tokenizer_revision,
        allow_patterns=TOKENIZER_PATTERNS,
        local_files_only=offline,
    )
    return AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )


def eos_token_ids(
    model_eos: int | list[int] | None, tokenizer_eos: int | None
) -> set[int]:
    """Keep both decoder EOS and chat terminators (which differ for Qwen)."""
    tokens = {model_eos} if isinstance(model_eos, int) else set(model_eos or [])
    if tokenizer_eos is not None:
        tokens.add(tokenizer_eos)
    if not tokens:
        raise ValueError("model/tokenizer must define an EOS token")
    return tokens


def encode_prompt(
    tokenizer: PreTrainedTokenizerBase,
    config: InferenceConfig,
    messages: tuple[Message, ...],
) -> EncodedPrompt:
    text = tokenizer.apply_chat_template(
        [message.model_dump() for message in messages],
        chat_template=tokenizer.get_chat_template(),
        tokenize=False,
        add_generation_prompt=True,
        **(
            {"enable_thinking": config.enable_thinking}
            if config.enable_thinking is not None
            else {}
        ),
    )
    if not isinstance(text, str):
        raise TypeError("chat template did not produce text")
    tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return EncodedPrompt(text=text, token_ids=tuple(tokens))


def decode_response(
    tokenizer: PreTrainedTokenizerBase,
    tokens: tuple[int, ...],
    eos_ids: set[int],
    thinking: bool | None,
) -> GeneratedResponse:
    ended = bool(tokens) and tokens[-1] in eos_ids
    content = tokens[:-1] if ended else tokens
    text = tokenizer.decode(
        content, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    reasoning_tokens = answer_tokens = None
    if thinking is True:
        boundary = tokenizer.encode("</think>", add_special_tokens=False)
        if len(boundary) != 1:
            raise ValueError("thinking requires a single-token Qwen closing delimiter")
        boundaries = content.count(boundary[0])
        if boundaries == 1:
            split = content.index(boundary[0])
            reasoning_tokens, answer_tokens = split, len(content) - split - 1
        elif boundaries == 0:
            reasoning_tokens, answer_tokens = len(content), 0
    return GeneratedResponse(
        text=text,
        token_ids=tokens,
        finish_reason="eos" if ended else "length",
        reasoning_tokens=reasoning_tokens,
        answer_tokens=answer_tokens,
    )


# Continue the given token IDs with at most this many generated tokens.
type Complete = Callable[[tuple[int, ...], int], tuple[int, ...]]

# Qwen's template renders the end of reasoning as "\n</think>\n\n" before the answer.
_FORCED_CLOSE = "\n</think>\n\n"


def generate_response(
    complete: Complete,
    tokenizer: PreTrainedTokenizerBase,
    config: InferenceConfig,
    prompt: EncodedPrompt,
    eos_ids: set[int],
) -> GeneratedResponse:
    """Generate one response, forcing the thinking boundary at its budget.

    Budget forcing (Muennighoff et al., 2025, "s1: Simple test-time scaling")
    stops unfinished reasoning by appending the closing delimiter; the model then
    answers from that prefix. Responses that close or end within the budget are
    returned unchanged. Inserted tokens count toward max_new_tokens.
    """
    budget = config.thinking_budget
    if budget is None:
        tokens = complete(prompt.token_ids, config.max_new_tokens)
        return decode_response(tokenizer, tokens, eos_ids, config.enable_thinking)
    thought = complete(prompt.token_ids, budget)
    (close,) = tokenizer.encode("</think>", add_special_tokens=False)
    if (thought and thought[-1] in eos_ids) or len(thought) < budget:
        return decode_response(tokenizer, thought, eos_ids, True)
    inserted: tuple[int, ...] = ()
    if close not in thought:
        inserted = tuple(tokenizer.encode(_FORCED_CLOSE, add_special_tokens=False))
        if inserted.count(close) != 1:
            raise ValueError("forced close must contain one closing delimiter")
    head = thought + inserted
    allowance = config.max_new_tokens - len(head)
    if allowance <= 0:
        raise ValueError("thinking budget leaves no answer allowance")
    rest = complete(prompt.token_ids + head, allowance)
    response = decode_response(tokenizer, head + rest, eos_ids, True)
    if not inserted:
        return response
    return response.model_copy(
        update={"forced_close_at": len(thought), "inserted_token_ids": inserted}
    )
