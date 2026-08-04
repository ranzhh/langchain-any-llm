"""Test ChatAnyLLM chat model."""

import msgpack
import pytest
from any_llm.types.completion import ChatCompletion
from langchain_core.messages import AIMessage

from langchain_anyllm import ChatAnyLLM


def _build_chat_completion(
    *,
    model: str = "openai:gpt-4o-mini",
    content: str = "hello",
) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "created": 0,
            "model": model,
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }
    )


class TestChatAnyLLM:
    """Test ChatAnyLLM class."""

    def test_initialization(self) -> None:
        """Test ChatAnyLLM initialization."""
        llm = ChatAnyLLM(model="gpt-4", model_kwargs={"temperature": 0.5})
        assert llm.model == "gpt-4"
        assert llm.model_kwargs["temperature"] == 0.5
        assert llm._llm_type == "anyllm-chat"

    def test_initialization_with_first_class_params(self) -> None:
        """Test ChatAnyLLM initialization with first-class parameters."""
        llm = ChatAnyLLM(
            model="gpt-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            api_key="test-key",
            api_base="https://custom.api.com",
        )
        assert llm.model == "gpt-4"
        assert llm.provider == "openai"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 100
        assert llm.top_p == 0.9
        assert llm.api_key == "test-key"
        assert llm.api_base == "https://custom.api.com"

    def test_default_params(self) -> None:
        """Test default parameters."""
        llm = ChatAnyLLM(
            model="gpt-3.5-turbo", model_kwargs={"temperature": 0.7, "max_tokens": 100}
        )
        params = llm._create_params()
        assert params["model"] == "gpt-3.5-turbo"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100

        # Test with additional params
        llm2 = ChatAnyLLM(model="gpt-4", model_kwargs={"n": 2})
        params2 = llm2._create_params()
        assert params2["n"] == 2

    def test_first_class_params_in_create_params(self) -> None:
        """Test that first-class parameters are included in _create_params."""
        llm = ChatAnyLLM(
            model="gpt-4",
            provider="openai",
            temperature=0.5,
            max_tokens=200,
            top_p=0.95,
            response_format={"type": "json_object"},
        )
        params = llm._create_params()
        assert params["model"] == "gpt-4"
        assert params["provider"] == "openai"
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 200
        assert params["top_p"] == 0.95
        assert params["response_format"] == {"type": "json_object"}

    def test_identifying_params_includes_provider(self) -> None:
        """Test that _identifying_params includes provider when set."""
        llm = ChatAnyLLM(model="gpt-4", provider="openai")
        params = llm._identifying_params
        assert params["model"] == "gpt-4"
        assert params["provider"] == "openai"

        # Without provider
        llm2 = ChatAnyLLM(model="openai:gpt-4")
        params2 = llm2._identifying_params
        assert params2["model"] == "openai:gpt-4"
        assert "provider" not in params2

    @pytest.mark.asyncio
    async def test_async_initialization(self) -> None:
        """Test async functionality exists."""
        llm = ChatAnyLLM(model="gpt-4")
        # Just verify the async methods exist
        assert hasattr(llm, "ainvoke")
        assert hasattr(llm, "astream")
        assert hasattr(llm, "_agenerate")

    def test_bind_tools(self) -> None:
        """Test binding tools to model."""
        llm = ChatAnyLLM(model="gpt-4")

        def dummy_tool(x: int) -> int:
            """Dummy tool."""
            return x * 2

        llm_with_tools = llm.bind_tools([dummy_tool])
        assert llm_with_tools is not None

    def test_with_structured_output(self) -> None:
        """Test structured output binding."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            """Test schema."""

            name: str
            age: int

        llm = ChatAnyLLM(model="gpt-4")
        structured_llm = llm.with_structured_output(TestSchema)
        assert structured_llm is not None

    def test_stream_options_requests_usage_by_default(self) -> None:
        """Streamed calls report usage without the caller having to opt in."""
        llm = ChatAnyLLM(model="openai:gpt-4o-mini")
        assert llm.stream_options == {"include_usage": True}

    def test_stream_options_can_be_overridden(self) -> None:
        """An explicitly passed stream_options still wins over the default."""
        llm = ChatAnyLLM(
            model="openai:gpt-4o-mini", stream_options={"include_usage": False}
        )
        assert llm.stream_options == {"include_usage": False}

    def test_create_chat_result_keeps_token_usage_msgpack_serializable(self) -> None:
        """Test token usage metadata stays serializable after LangChain merges it."""
        llm = ChatAnyLLM(model="openai:gpt-4o-mini")
        result = llm._create_chat_result(_build_chat_completion())
        assert result.llm_output is not None
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
        message.response_metadata = {**result.llm_output, **message.response_metadata}

        assert result.llm_output == {
            "token_usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "completion_tokens_details": None,
                "prompt_tokens_details": None,
            },
            "model": "openai:gpt-4o-mini",
        }
        assert message.usage_metadata == {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
        }
        msgpack.packb(message.model_dump())

    @pytest.mark.parametrize(
        ("model", "provider", "expected_provider", "expected_model"),
        [
            ("openai:gpt-4o-mini", None, "openai", "gpt-4o-mini"),
            ("gpt-4o-mini", "openai", "openai", "gpt-4o-mini"),
            ("openai:gpt-4o-mini", "openai", "openai", "gpt-4o-mini"),
            ("gemini:gemini-2.0-flash", None, "google_genai", "gemini-2.0-flash"),
            ("llama3.2:3b", "ollama", "ollama", "llama3.2:3b"),
        ],
    )
    def test_get_ls_params_normalizes_provider_and_model(
        self,
        model: str,
        provider: str | None,
        expected_provider: str,
        expected_model: str,
    ) -> None:
        """Test LangSmith metadata matches AnyLLM provider/model semantics."""
        llm = ChatAnyLLM(
            model=model,
            provider=provider,
            temperature=0.5,
            max_tokens=64,
        )

        ls_params = llm._get_ls_params(stop=["DONE"])

        assert ls_params["ls_provider"] == expected_provider
        assert ls_params["ls_model_name"] == expected_model
        assert ls_params["ls_model_type"] == "chat"
        assert ls_params["ls_temperature"] == 0.5
        assert ls_params["ls_max_tokens"] == 64
        assert ls_params["ls_stop"] == ["DONE"]
