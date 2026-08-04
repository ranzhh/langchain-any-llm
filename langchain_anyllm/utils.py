"""Utility functions for message conversion."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from any_llm.types.completion import ChoiceDelta
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)

logger = logging.getLogger(__name__)


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: Dictionary containing message data with 'role' and 'content' keys

    Returns:
        Appropriate BaseMessage subclass based on role
    """
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    if role == "assistant":
        # Handle content - can be string, None, or missing
        # When tool calls are present, content might be None
        content = _dict.get("content") or ""

        additional_kwargs = {}
        if _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(_dict["function_call"])

        tool_calls = []
        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]
            for tool_call in _dict["tool_calls"]:
                try:
                    tool_calls.append(
                        ToolCall(
                            name=tool_call["function"]["name"],
                            args=json.loads(tool_call["function"]["arguments"]),
                            id=tool_call.get("id"),
                        )
                    )
                except (KeyError, json.JSONDecodeError) as e:
                    logger.debug(f"Skipping malformed tool call: {e}")
                    continue

        return AIMessage(
            content=content, additional_kwargs=additional_kwargs, tool_calls=tool_calls
        )
    if role == "system":
        return SystemMessage(content=_dict["content"])
    if role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    if role == "tool":
        return ToolMessage(content=_dict["content"], tool_call_id=_dict["tool_call_id"])
    if not role:
        return ChatMessage(content=_dict["content"], role="unknown")
    return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    delta: ChoiceDelta | dict[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a streaming delta to a LangChain message chunk.

    Args:
        delta: Delta object from streaming response
        default_class: Default message chunk class to use

    Returns:
        Appropriate BaseMessageChunk subclass
    """
    # Initialize common variables that will be used for all paths
    additional_kwargs: dict[str, Any] = {}
    tool_call_chunks: list[ToolCallChunk] = []

    if isinstance(delta, dict):
        # Handle dictionary-style delta
        role = delta.get("role")
        content = delta.get("content") or ""
        function_call = delta.get("function_call")
        raw_tool_calls = delta.get("tool_calls")

        # Build additional_kwargs from dict delta
        if function_call:
            additional_kwargs["function_call"] = function_call

        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            additional_kwargs["reasoning_content"] = reasoning_content

        # Handle provider-specific fields
        provider_specific_fields = delta.get("provider_specific_fields")
        if not provider_specific_fields:
            provider_specific_fields = delta.get("vertex_ai_grounding_metadata")
        if provider_specific_fields:
            additional_kwargs["provider_specific_fields"] = provider_specific_fields

        # Build tool_call_chunks from dict delta
        if raw_tool_calls:
            for rtc in raw_tool_calls:
                try:
                    is_dict = isinstance(rtc, dict)
                    func = (
                        rtc.get("function")
                        if is_dict
                        else getattr(rtc, "function", None)
                    )
                    if func:
                        func_is_dict = isinstance(func, dict)
                        name = (
                            func.get("name", "")
                            if func_is_dict
                            else getattr(func, "name", "")
                        )
                        args = (
                            func.get("arguments", "")
                            if func_is_dict
                            else getattr(func, "arguments", "")
                        )
                        rtc_id = rtc.get("id") if is_dict else getattr(rtc, "id", None)
                        rtc_index = (
                            rtc.get("index") if is_dict else getattr(rtc, "index", None)
                        )
                        tool_call_chunks.append(
                            ToolCallChunk(
                                name=name, args=args, id=rtc_id, index=rtc_index
                            )
                        )
                except (KeyError, AttributeError, TypeError) as e:
                    logger.debug(f"Skipping malformed tool call chunk: {e}")
                    continue
    else:
        # Handle object-style delta (ChoiceDelta)
        role = delta.role or ""
        content = delta.content or ""

        # Build additional_kwargs from object delta
        if delta.function_call:
            additional_kwargs["function_call"] = dict(delta.function_call)

        # Handle reasoning attribute (e.g., from o1 models)
        reasoning = getattr(delta, "reasoning", None)
        if reasoning:
            reasoning_content = (
                reasoning.content
                if hasattr(reasoning, "content")
                else reasoning.get("content")
            )
            if reasoning_content:
                additional_kwargs["reasoning_content"] = reasoning_content

        # Build tool_call_chunks from object delta
        if raw_tool_calls := delta.tool_calls:
            # Store dicts; the raw objects are not reliably serializable
            additional_kwargs["tool_calls"] = [
                rtc.model_dump() if hasattr(rtc, "model_dump") else rtc
                for rtc in raw_tool_calls
            ]
            try:
                tool_call_chunks = [
                    ToolCallChunk(
                        name=rtc.function.name if rtc.function else "",
                        args=rtc.function.arguments if rtc.function else "",
                        id=rtc.id,
                        index=rtc.index,
                    )
                    for rtc in raw_tool_calls
                    if rtc.function
                ]
            except KeyError:
                pass

    # Return appropriate message chunk based on role
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if default_class == FunctionMessageChunk:
        if isinstance(delta, dict):
            func_call = delta.get("function_call")
            if func_call:
                fc_is_dict = isinstance(func_call, dict)
                fc_args = (
                    func_call.get("arguments", "")
                    if fc_is_dict
                    else getattr(func_call, "arguments", "") or ""
                )
                fc_name = (
                    func_call.get("name", "")
                    if fc_is_dict
                    else getattr(func_call, "name", "") or ""
                )
                return FunctionMessageChunk(content=fc_args, name=fc_name)
        elif hasattr(delta, "function_call") and delta.function_call:
            return FunctionMessageChunk(
                content=delta.function_call.arguments or "",
                name=delta.function_call.name or "",
            )
    if role == "tool" or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    return default_class(content=content)  # type: ignore[call-arg]


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """Convert a LangChain ToolCall to OpenAI tool call format.

    Args:
        tool_call: LangChain ToolCall object

    Returns:
        Dictionary in OpenAI tool call format
    """
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _filter_anthropic_content_blocks(content: Any) -> Any:
    """Filter Anthropic-style content blocks to OpenAI-compatible format.

    Removes tool_use blocks since they're represented separately in tool_calls.
    OpenAI supports: 'text', 'image_url', 'input_audio', 'refusal', 'audio', 'file'.

    Args:
        content: Message content (can be string, list, or other)

    Returns:
        Filtered content compatible with OpenAI API
    """
    # If content is not a list, return as-is
    if not isinstance(content, list):
        return content

    # Filter out tool_use blocks (and any other unsupported types)
    # Keep only OpenAI-compatible content types
    openai_compatible_types = {
        "text",
        "image_url",
        "input_audio",
        "refusal",
        "audio",
        "file",
    }

    filtered_content = [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") in openai_compatible_types
    ]

    # If no blocks remain, return empty string
    if not filtered_content:
        return ""

    # If only one text block remains, simplify to just the text string
    if len(filtered_content) == 1 and filtered_content[0].get("type") == "text":
        return filtered_content[0].get("text", "")

    # Otherwise return the filtered list
    return filtered_content


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a dictionary.

    Args:
        message: LangChain message object

    Returns:
        Dictionary representation of the message

    Raises:
        ValueError: If message type is unknown
    """
    message_dict: dict[str, Any] = {
        "content": _filter_anthropic_content_blocks(message.content)
    }
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
        message_dict["name"] = message.name
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        error_message = f"Got unknown type {message}"
        raise ValueError(error_message)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict
