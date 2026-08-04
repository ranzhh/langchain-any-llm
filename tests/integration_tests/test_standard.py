"""Standard integration tests for ChatAnyLLM.

This module contains standard integration tests that verify ChatAnyLLM
conforms to the LangChain ChatModel interface.
"""

import os
from typing import Type

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_anyllm import ChatAnyLLM


class TestChatAnyLLMStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatAnyLLM]:
        return ChatAnyLLM

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "openai:gpt-4o-mini",
            "temperature": 0,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    @property
    def supports_video_inputs(self) -> bool:
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        return False

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_structured_output(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @property
    def supports_parallel_tool_calls(self) -> bool:
        return True


# Skip tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set. "
    "These tests require a valid API key to run.",
)
