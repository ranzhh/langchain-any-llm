"""AnyLLM chat model integration for LangChain.

This module provides a LangChain-compatible chat model wrapper for AnyLLM,
enabling seamless integration with LangChain's ecosystem.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Sequence,
)

from any_llm import acompletion, completion
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthExceededError,
    InvalidRequestError,
    ModelNotFoundError,
    RateLimitError,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.base import LangSmithParams
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from langchain_anyllm.utils import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
)

logger = logging.getLogger(__name__)

_LS_PROVIDER_ALIASES = {
    "gemini": "google_genai",
    "vertexai": "google_vertexai",
}


class ChatAnyLLM(BaseChatModel):
    """Chat model that uses the AnyLLM API.

    This class provides a LangChain-compatible interface to any-llm, which supports
    multiple LLM providers (OpenAI, Anthropic, Google, local models, etc.) through
    a unified API.

    Example:
        .. code-block:: python

            from langchain_anyllm import ChatAnyLLM

            # Using model string with provider prefix
            llm = ChatAnyLLM(model="openai:gpt-4")

            # Or using separate provider parameter
            llm = ChatAnyLLM(model="gpt-4", provider="openai")

            response = llm.invoke("Hello, how are you?")

    Attributes:
        model: The model identifier. Can include provider prefix (e.g., "openai:gpt-4")
            or be used with separate provider parameter.
        provider: Optional provider name. If not specified, extracted from model string.
        api_key: API key for the provider. If not set, uses environment variable.
        api_base: Custom API base URL for the provider.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling parameter.
        response_format: Response format specification. Use {"type": "json_object"}
            for JSON mode.
        model_kwargs: Additional model parameters passed to the API.
    """

    model: str
    provider: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    response_format: dict[str, Any] | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Streamed responses omit usage unless it is asked for
    stream_options: dict[str, Any] | None = {"include_usage": True}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from the model.

        Args:
            messages: List of messages in the conversation.
            stop: Optional list of stop sequences.
            run_manager: Optional callback manager for the run.
            stream: Whether to stream the response.
            **kwargs: Additional parameters passed to the model.

        Returns:
            ChatResult containing the model's response.

        Raises:
            ValueError: If the response type is unexpected.
        """
        if stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts = [_convert_message_to_dict(m) for m in messages]
        logger.debug(f"Message dicts: {message_dicts}")
        params = self._create_params(stop, **kwargs)
        response = self._call_completion(message_dicts, params)
        return self._create_chat_result(response)

    def _call_completion(
        self, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> ChatCompletion:
        """Call the any-llm completion API with error handling.

        Args:
            messages: List of message dictionaries.
            params: Parameters for the completion call.

        Returns:
            ChatCompletion response from the API.

        Raises:
            ValueError: If authentication fails or API key is missing.
            RuntimeError: If the model is not found or provider error occurs.
        """
        try:
            response = completion(messages=messages, **params)  # type: ignore[arg-type]
        except AuthenticationError as e:
            raise ValueError(f"Authentication failed: {e}") from e
        except ModelNotFoundError as e:
            raise ValueError(f"Model not found: {e}") from e
        except RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {e}") from e
        except ContextLengthExceededError as e:
            raise ValueError(f"Context length exceeded: {e}") from e
        except ContentFilterError as e:
            raise ValueError(f"Content filtered: {e}") from e
        except InvalidRequestError as e:
            raise ValueError(f"Invalid request: {e}") from e
        except AnyLLMError as e:
            raise RuntimeError(f"AnyLLM error: {e}") from e

        if not isinstance(response, ChatCompletion):
            error_message = f"Expected ChatCompletion, got {type(response)}"
            raise ValueError(error_message)
        return response

    async def _acall_completion(
        self, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> ChatCompletion:
        """Call the any-llm async completion API with error handling.

        Args:
            messages: List of message dictionaries.
            params: Parameters for the completion call.

        Returns:
            ChatCompletion response from the API.

        Raises:
            ValueError: If authentication fails or API key is missing.
            RuntimeError: If the model is not found or provider error occurs.
        """
        try:
            response = await acompletion(messages=messages, **params)  # type: ignore[arg-type]
        except AuthenticationError as e:
            raise ValueError(f"Authentication failed: {e}") from e
        except ModelNotFoundError as e:
            raise ValueError(f"Model not found: {e}") from e
        except RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {e}") from e
        except ContextLengthExceededError as e:
            raise ValueError(f"Context length exceeded: {e}") from e
        except ContentFilterError as e:
            raise ValueError(f"Content filtered: {e}") from e
        except InvalidRequestError as e:
            raise ValueError(f"Invalid request: {e}") from e
        except AnyLLMError as e:
            raise RuntimeError(f"AnyLLM error: {e}") from e

        if not isinstance(response, ChatCompletion):
            error_message = f"Expected ChatCompletion, got {type(response)}"
            raise ValueError(error_message)
        return response

    def _extract_usage_metadata(
        self, usage: dict[str, Any] | None
    ) -> UsageMetadata | None:
        """Extract usage metadata from a usage dictionary.

        Reasoning and cached tokens are subsets of the completion and prompt counts
        respectively, which is the same convention LangChain uses for the detail
        fields, so both are reported without adjusting the totals.

        Args:
            usage: Dictionary containing usage information.

        Returns:
            UsageMetadata object or None if usage is not available, carrying detail
            fields only for the categories the provider reported.
        """
        if not usage:
            return None
        metadata = UsageMetadata(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = completion_details.get("reasoning_tokens")
        if reasoning_tokens:
            metadata["output_token_details"] = OutputTokenDetails(
                reasoning=reasoning_tokens
            )
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = prompt_details.get("cached_tokens")
        if cached_tokens:
            metadata["input_token_details"] = InputTokenDetails(
                cache_read=cached_tokens
            )
        return metadata

    def _create_chat_result(self, response: ChatCompletion) -> ChatResult:
        """Create a ChatResult from an API response.

        Args:
            response: ChatCompletion response from the API.

        Returns:
            ChatResult containing the parsed response.
        """
        resp_dict = response.model_dump()

        generations = []
        token_usage = resp_dict.get("usage")
        usage_metadata = self._extract_usage_metadata(token_usage)
        for res in resp_dict["choices"]:
            message = _convert_dict_to_message(res["message"])
            if isinstance(message, AIMessage) and usage_metadata:
                message.response_metadata = {"model_name": self.model}
                message.usage_metadata = usage_metadata
            gen = ChatGeneration(
                message=message,
                generation_info={"finish_reason": res.get("finish_reason")},
            )
            generations.append(gen)

        llm_output = {
            "token_usage": token_usage,
            "model": self.model,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _is_anthropic_model(self) -> bool:
        """Check if the model is an Anthropic model."""
        return self.model.startswith("anthropic:") or "claude" in self.model.lower()

    def _create_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Create parameters dictionary for the API call.

        Args:
            stop: Optional list of stop sequences.
            **kwargs: Additional parameters to include.

        Returns:
            Dictionary of parameters for the completion call.

        Raises:
            ValueError: If stop is specified in both input and model_kwargs.
        """
        params: dict[str, Any] = {
            "model": self.model,
            **self.model_kwargs,
        }

        # Add optional parameters only if set
        if self.provider is not None:
            params["provider"] = self.provider
        if self.api_key is not None:
            params["api_key"] = self.api_key
        if self.api_base is not None:
            params["api_base"] = self.api_base
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.response_format is not None:
            params["response_format"] = self.response_format

        if stop is not None:
            if "stop" in params or "stop_sequences" in params:
                error_message = "`stop` found in both the input and default params."
                raise ValueError(error_message)
            # Anthropic uses stop_sequences instead of stop
            if self._is_anthropic_model():
                params["stop_sequences"] = stop
            else:
                params["stop"] = stop

        # Translate LangChain tool_choice to OpenAI-compatible values
        # Only include tool_choice if tools are present
        if "tool_choice" in kwargs and "tools" in kwargs:
            tool_choice = kwargs["tool_choice"]
            if tool_choice == "any":
                # LangChain uses 'any', OpenAI uses 'required'
                params["tool_choice"] = "required"
            elif tool_choice is True:
                params["tool_choice"] = "required"
            elif tool_choice is False:
                params["tool_choice"] = "none"
            elif isinstance(tool_choice, str) and tool_choice not in [
                "none",
                "auto",
                "required",
            ]:
                # If it's a string that's not a standard value,
                # treat it as a function name
                params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }
            else:
                # Pass through dicts and standard string values
                params["tool_choice"] = tool_choice

        # Pass through all kwargs except our special handling
        for key, value in kwargs.items():
            if key not in ["tool_choice"]:
                params[key] = value

        return params

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat responses from the model.

        Args:
            messages: List of messages in the conversation.
            stop: Optional list of stop sequences.
            run_manager: Optional callback manager for the run.
            **kwargs: Additional parameters passed to the model.

        Yields:
            ChatGenerationChunk objects for each streamed response chunk.
        """
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = self._create_params(stop, **kwargs)
        params["stream"] = True

        # Anthropic doesn't support stream_options
        if not self._is_anthropic_model():
            if "stream_options" not in params and self.stream_options:
                params["stream_options"] = self.stream_options

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk

        try:
            result = completion(messages=message_dicts, **params)  # type: ignore[arg-type]
        except AuthenticationError as e:
            raise ValueError(f"Authentication failed: {e}") from e
        except ModelNotFoundError as e:
            raise ValueError(f"Model not found: {e}") from e
        except RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {e}") from e
        except AnyLLMError as e:
            raise RuntimeError(f"AnyLLM error: {e}") from e

        if not isinstance(result, Iterator):
            error_message = f"Expected Iterator, got {type(result)}"
            raise ValueError(error_message)

        for chunk_item in result:
            chunk_dict: dict[str, Any] = chunk_item.model_dump()

            # Handle usage-only chunk (final chunk with empty choices but usage data)
            if len(chunk_dict["choices"]) == 0:
                usage_metadata = self._extract_usage_metadata(chunk_dict.get("usage"))
                if usage_metadata:
                    usage_chunk = AIMessageChunk(
                        content="",
                        response_metadata={"model_name": self.model},
                        usage_metadata=usage_metadata,
                    )
                    yield ChatGenerationChunk(message=usage_chunk)
                continue

            choice = chunk_dict["choices"][0]
            delta = choice["delta"]

            message_chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)

            # Only set usage_metadata on the final chunk (when finish_reason is set)
            finish_reason = choice.get("finish_reason")
            if finish_reason and chunk_dict.get("usage"):
                if isinstance(message_chunk, AIMessageChunk):
                    message_chunk.usage_metadata = self._extract_usage_metadata(
                        chunk_dict["usage"]
                    )
                    message_chunk.response_metadata = {"model_name": self.model}

            default_chunk_class = message_chunk.__class__
            cg_chunk = ChatGenerationChunk(message=message_chunk)
            if run_manager:
                content = message_chunk.content
                if isinstance(content, str):
                    run_manager.on_llm_new_token(content, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat responses from the model.

        Args:
            messages: List of messages in the conversation.
            stop: Optional list of stop sequences.
            run_manager: Optional callback manager for the run.
            **kwargs: Additional parameters passed to the model.

        Yields:
            ChatGenerationChunk objects for each streamed response chunk.
        """
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = self._create_params(stop, **kwargs)
        params["stream"] = True

        # Anthropic doesn't support stream_options
        if not self._is_anthropic_model():
            if "stream_options" not in params and self.stream_options:
                params["stream_options"] = self.stream_options

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk

        try:
            result = await acompletion(messages=message_dicts, **params)  # type: ignore[arg-type]
        except AuthenticationError as e:
            raise ValueError(f"Authentication failed: {e}") from e
        except ModelNotFoundError as e:
            raise ValueError(f"Model not found: {e}") from e
        except RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {e}") from e
        except AnyLLMError as e:
            raise RuntimeError(f"AnyLLM error: {e}") from e

        if not isinstance(result, AsyncIterator):
            error_message = f"Expected AsyncIterator, got {type(result)}"
            raise ValueError(error_message)

        async for stream_chunk in result:
            if not isinstance(stream_chunk, ChatCompletionChunk):
                error_message = "Unexpected chunk type"
                raise ValueError(error_message)

            # Handle usage-only chunk (final chunk with empty choices but usage data)
            if len(stream_chunk.choices) == 0:
                if hasattr(stream_chunk, "usage") and stream_chunk.usage:
                    usage = stream_chunk.usage.model_dump()
                    usage_metadata = self._extract_usage_metadata(usage)
                    if usage_metadata:
                        usage_chunk = AIMessageChunk(
                            content="",
                            response_metadata={"model_name": self.model},
                            usage_metadata=usage_metadata,
                        )
                        yield ChatGenerationChunk(message=usage_chunk)
                continue

            for choice in stream_chunk.choices:
                delta = choice.delta
                message_chunk = _convert_delta_to_message_chunk(
                    delta, default_chunk_class
                )

                # Only set usage_metadata on the final chunk (when finish_reason is set)
                if choice.finish_reason:
                    if hasattr(stream_chunk, "usage") and stream_chunk.usage:
                        if isinstance(message_chunk, AIMessageChunk):
                            usage = stream_chunk.usage.model_dump()
                            message_chunk.usage_metadata = self._extract_usage_metadata(
                                usage
                            )
                            message_chunk.response_metadata = {"model_name": self.model}

                default_chunk_class = message_chunk.__class__
                cg_chunk = ChatGenerationChunk(message=message_chunk)
                if run_manager:
                    content = message_chunk.content
                    if isinstance(content, str):
                        await run_manager.on_llm_new_token(content, chunk=cg_chunk)
                yield cg_chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a chat response from the model.

        Args:
            messages: List of messages in the conversation.
            stop: Optional list of stop sequences.
            run_manager: Optional callback manager for the run.
            stream: Whether to stream the response.
            **kwargs: Additional parameters passed to the model.

        Returns:
            ChatResult containing the model's response.
        """
        should_stream = stream if stream is not None else False
        if should_stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = self._create_params(stop, **kwargs)
        response = await self._acall_completion(message_dicts, params)
        return self._create_chat_result(response)

    def bind_tools(
        self,
        tools: Sequence[
            dict[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool
        ],
        tool_choice: dict[str, Any] | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: Sequence of tools to bind. Can be dictionaries, Pydantic models,
                callables, or BaseTool instances.
            tool_choice: How the model should choose tools. Can be:
                - None: Model decides whether to use tools
                - "auto": Model decides whether to use tools
                - "required" or "any" or True: Model must use a tool
                - "none" or False: Model must not use tools
                - str: Name of specific tool to use
                - dict: Full tool_choice specification
            **kwargs: Additional parameters passed to the model.

        Returns:
            A Runnable that will use the bound tools.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters for this model."""
        params: dict[str, Any] = {"model": self.model}
        if self.provider:
            params["provider"] = self.provider
        params.update(self.model_kwargs)
        return params

    def _get_ls_provider_and_model(
        self, *, provider: str | None = None, model: str | None = None
    ) -> tuple[str | None, str]:
        """Normalize provider/model names for LangSmith metadata."""
        provider = provider or self.provider
        model = model or self.model

        if provider:
            provider = provider.lower()
            prefixed_provider = f"{provider}:"
            if model.startswith(prefixed_provider):
                return provider, model[len(prefixed_provider) :]
            return provider, model

        if ":" in model:
            parsed_provider, _, parsed_model = model.partition(":")
            return parsed_provider.lower(), parsed_model

        return None, model

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Return LangSmith tracing params with normalized AnyLLM metadata."""
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        provider, model = self._get_ls_provider_and_model(
            provider=kwargs.get("provider"),
            model=kwargs.get("model"),
        )
        if provider:
            ls_params["ls_provider"] = _LS_PROVIDER_ALIASES.get(provider, provider)
        ls_params["ls_model_name"] = model
        return ls_params

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "anyllm-chat"
