"""Pinned chat encoding and lossless response decoding shared by LLM backends."""

from transformers import PreTrainedTokenizerBase

from manabot.llm.schema import (
    EncodedPrompt,
    GeneratedResponse,
    InferenceConfig,
    Message,
)


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
        if content.count(boundary[0]) == 1:
            split = content.index(boundary[0])
            reasoning_tokens, answer_tokens = split, len(content) - split - 1
        elif boundary[0] not in content:
            reasoning_tokens, answer_tokens = len(content), 0
    return GeneratedResponse(
        text=text,
        token_ids=tokens,
        finish_reason="eos" if ended else "length",
        reasoning_tokens=reasoning_tokens,
        answer_tokens=answer_tokens,
    )
