# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: Apache-2.0
"""mlx-lm-facing adapter for DeepSeek V4's DSML chat template.

The encoding itself lives in the sibling module ``encoding_dsv4`` — a
verbatim vendor copy of the reference implementation DeepSeek ships
alongside the model weights on HuggingFace. We re-export the encoding
module's symbols so callers that imported from ``chat_template_v4``
keep working, and we add a thin ``apply_chat_template`` that bridges
mlx-lm's call signature to ``encoding_dsv4.encode_messages``.

Historical note. An earlier version of this file was a near-verbatim
copy of mlx-lm's V3.2 chat template with only the outer DSML marker
name renamed (function_calls → tool_calls), citing vllm's
``DeepSeekV4ToolParser`` as the authority. That authority only
specified what to *parse* — it said nothing about the prompt-side
framing the model was trained on. The shipped ``encoding_dsv4.py``
diverges from the V3.2 derivation on terminology ("tools" not
"functions"), schema wrapping (bare under ``### Available Tool
Schemas`` not inside ``<functions>``), tool-output marker
(``<tool_result>`` not ``<function_results><result>``), thinking-mode
imperative phrasing, and several other places. omlx now defers to the
publisher's spec wholesale.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import encoding_dsv4 as _enc
# Re-export the encoding module surface so existing callers keep their
# imports. Names follow the upstream module verbatim; the adapter
# layer only adds ``apply_chat_template``.
from .encoding_dsv4 import (  # noqa: F401
    ASSISTANT_SP_TOKEN,
    DS_TASK_SP_TOKENS,
    LATEST_REMINDER_SP_TOKEN,
    REASONING_EFFORT_MAX,
    TOOLS_TEMPLATE,
    USER_SP_TOKEN,
    VALID_TASKS,
    assistant_msg_template,
    assistant_msg_wo_eos_template,
    bos_token,
    decode_dsml_to_arguments,
    dsml_token,
    encode_arguments_to_dsml,
    encode_messages,
    eos_token,
    find_last_user_index,
    merge_tool_messages,
    parse_message_from_completion_text,
    parse_tool_calls,
    render_message,
    render_tools,
    response_format_template,
    sort_tool_results_by_call_order,
    system_msg_template,
    thinking_end_token,
    thinking_start_token,
    thinking_template,
    to_json,
    tool_call_template,
    tool_calls_block_name,
    tool_calls_from_openai_format,
    tool_calls_template,
    tool_calls_to_openai_format,
    tool_output_template,
    tools_from_openai_format,
    user_msg_template,
)


_ENCODE_ACCEPTED_KWARGS = {
    "thinking_mode",
    "context",
    "drop_thinking",
    "add_default_bos_token",
    "reasoning_effort",
}


def _normalize_tool_call_arguments(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """JSON-stringify dict-typed ``tool_call["function"]["arguments"]``.

    The omlx Anthropic adapter (``api/anthropic_utils.py``) decodes
    Claude's ``input`` field into a dict before storing it on assistant
    messages. The shipped ``encode_arguments_to_dsml`` only knows the
    OpenAI JSON-string convention and would otherwise fall back to
    wrapping the entire dict under a single ``arguments`` parameter,
    breaking multi-turn history. Normalize at the adapter boundary so
    the vendored encoding module stays byte-identical to upstream.
    """
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            normalized.append(msg)
            continue
        new_calls = []
        for tc in msg["tool_calls"]:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, dict):
                tc = {
                    **tc,
                    "function": {
                        **fn,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            new_calls.append(tc)
        normalized.append({**msg, "tool_calls": new_calls})
    return normalized


def _inject_top_level_tools(
    messages: List[Dict[str, Any]], tools: Any
) -> List[Dict[str, Any]]:
    """Move a top-level ``tools`` kwarg onto the first system / developer
    message (or synthesise one) so ``render_message`` emits the DSML
    tools block. mlx-lm and the omlx server pass ``tools`` as a kwarg
    to ``apply_chat_template``, but ``encoding_dsv4`` reads it off the
    message.
    """
    if not tools:
        return messages
    if messages and messages[0].get("role") in ("system", "developer"):
        if messages[0].get("tools"):
            return messages
        return [{**messages[0], "tools": tools}, *messages[1:]]
    return [{"role": "system", "content": "", "tools": tools}, *messages]


def apply_chat_template(
    messages: List[Dict[str, Any]],
    continue_final_message: bool = False,
    add_generation_prompt: bool = False,
    **kwargs: Any,
) -> str:
    """mlx-lm-facing entry point. Bridges ``apply_chat_template``'s
    signature to ``encoding_dsv4.encode_messages`` and applies omlx
    boundary fixes (dict-typed tool-call arguments, top-level ``tools``
    kwarg) without modifying the vendored encoding module.
    """
    if continue_final_message and add_generation_prompt:
        raise ValueError(
            "Only one of continue_final_message or add_generation_prompt can be True"
        )

    if "enable_thinking" in kwargs and "thinking_mode" not in kwargs:
        kwargs["thinking_mode"] = (
            "thinking" if kwargs.pop("enable_thinking") else "chat"
        )
    else:
        kwargs.pop("enable_thinking", None)

    messages = _normalize_tool_call_arguments(list(messages))
    messages = _inject_top_level_tools(messages, kwargs.pop("tools", None))

    if (
        continue_final_message
        and messages
        and messages[-1].get("role") == "assistant"
    ):
        messages = [*messages[:-1], {**messages[-1], "wo_eos": True}]

    kwargs = {k: v for k, v in kwargs.items() if k in _ENCODE_ACCEPTED_KWARGS}
    kwargs.setdefault("thinking_mode", "thinking")
    # V4Cache is non-block-sliceable: SSD prefix-cache entries store atomic
    # full-state snapshots keyed by the exact input-token sequence that
    # produced them. drop_thinking=True re-renders prior assistant turns
    # without their reasoning_content, so turn 2's encoded prompt drifts
    # away from turn 1's cached state at the first reasoning token and
    # never hits the prefix cache. Force False at the adapter so cross-
    # turn cache reuse works. (Caller can still override by passing
    # drop_thinking=True explicitly.)
    kwargs.setdefault("drop_thinking", False)

    out = _enc.encode_messages(messages, **kwargs)

    # encoding_dsv4 unconditionally appends ``<｜Assistant｜><think>`` (or
    # ``</think>``) after a trailing user/developer message. mlx-lm
    # callers that pass ``add_generation_prompt=False`` (e.g. for SFT
    # data preparation) want history without the assistant primer.
    if not add_generation_prompt and messages and messages[-1].get("role") in (
        "user",
        "developer",
    ):
        out = out.removesuffix(
            _enc.ASSISTANT_SP_TOKEN + _enc.thinking_start_token
        )
        out = out.removesuffix(
            _enc.ASSISTANT_SP_TOKEN + _enc.thinking_end_token
        )

    return out
