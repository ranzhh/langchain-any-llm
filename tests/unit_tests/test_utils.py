"""Test utility functions."""

import msgpack
from any_llm.types.completion import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from langchain_anyllm.utils import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _lc_tool_call_to_openai_tool_call,
)


class TestMessageConversion:
    """Test message conversion utilities."""

    def test_convert_human_message_to_dict(self) -> None:
        """Test converting HumanMessage to dict."""
        message = HumanMessage(content="Hello")
        result = _convert_message_to_dict(message)
        assert result == {"role": "user", "content": "Hello"}

    def test_convert_ai_message_to_dict(self) -> None:
        """Test converting AIMessage to dict."""
        message = AIMessage(content="Hi there")
        result = _convert_message_to_dict(message)
        assert result == {"role": "assistant", "content": "Hi there"}

    def test_convert_system_message_to_dict(self) -> None:
        """Test converting SystemMessage to dict."""
        message = SystemMessage(content="You are helpful")
        result = _convert_message_to_dict(message)
        assert result == {"role": "system", "content": "You are helpful"}

    def test_convert_dict_to_human_message(self) -> None:
        """Test converting dict to HumanMessage."""
        message_dict = {"role": "user", "content": "Hello"}
        result = _convert_dict_to_message(message_dict)
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello"

    def test_convert_dict_to_ai_message(self) -> None:
        """Test converting dict to AIMessage."""
        message_dict = {"role": "assistant", "content": "Hi"}
        result = _convert_dict_to_message(message_dict)
        assert isinstance(result, AIMessage)
        assert result.content == "Hi"

    def test_convert_dict_to_system_message(self) -> None:
        """Test converting dict to SystemMessage."""
        message_dict = {"role": "system", "content": "Be helpful"}
        result = _convert_dict_to_message(message_dict)
        assert isinstance(result, SystemMessage)
        assert result.content == "Be helpful"

    def test_convert_ai_message_with_tool_calls(self) -> None:
        """Test converting AIMessage with tool calls."""
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "name": "get_weather",
                    "args": {"location": "Paris"},
                }
            ],
        )
        result = _convert_message_to_dict(message)
        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"

    def test_lc_tool_call_to_openai_format(self) -> None:
        """Test converting LangChain tool call to OpenAI format."""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            args={"location": "Paris", "unit": "celsius"},
        )
        result = _lc_tool_call_to_openai_tool_call(tool_call)
        assert result["type"] == "function"
        assert result["id"] == "call_123"
        assert result["function"]["name"] == "get_weather"
        assert "location" in result["function"]["arguments"]

    def test_convert_tool_message(self) -> None:
        """Test converting ToolMessage."""
        message = ToolMessage(content="Temperature is 20C", tool_call_id="call_123")
        result = _convert_message_to_dict(message)
        assert result["role"] == "tool"
        assert result["content"] == "Temperature is 20C"
        assert result["tool_call_id"] == "call_123"

    def test_convert_dict_with_empty_content(self) -> None:
        """Test converting dict with None/empty content."""
        message_dict = {"role": "assistant", "content": None}
        result = _convert_dict_to_message(message_dict)
        assert isinstance(result, AIMessage)
        assert result.content == ""


class TestDeltaConversion:
    """Test streaming delta conversion utilities."""

    def test_delta_tool_calls_are_stored_as_plain_dicts(self) -> None:
        """Raw ChoiceDeltaToolCall objects must not reach additional_kwargs.

        Their pydantic serializer is a lazy MockValSer that has not been built
        yet, so a later model_dump() of the message raises TypeError and takes
        any tracing callback or checkpoint write down with it. msgpack packing
        stands in for that downstream serialization here.
        """
        delta = ChoiceDelta(
            role="assistant",
            content="",
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="call_1",
                    type="function",
                    function=ChoiceDeltaToolCallFunction(
                        name="get_weather", arguments='{"city": "Rome"}'
                    ),
                )
            ],
        )

        chunk = _convert_delta_to_message_chunk(delta, AIMessageChunk)
        assert isinstance(chunk, AIMessageChunk)

        stored = chunk.additional_kwargs["tool_calls"]
        assert all(isinstance(tc, dict) for tc in stored)
        assert msgpack.packb(stored) is not None
        # the parsed chunks still carry the call through
        assert chunk.tool_call_chunks[0]["name"] == "get_weather"
