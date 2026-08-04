# langchain-anyllm

**One interface for every LLM.**

This integration enables you to use [any-llm's](https://github.com/mozilla-ai/any-llm) unified interface (supporting OpenAI, Anthropic, Gemini, local models, and more) as a standard LangChain `ChatModel`. See all `any-llm` supported providers [here](https://mozilla-ai.github.io/any-llm/providers/)

No need to rewrite your provider-specific adapter code every time you want to test a new model. Switch between OpenAI, Anthropic, Gemini, and local models (via Ollama/LocalAI) just by changing a string.

This is a fork of [mozilla-ai/langchain-any-llm](https://github.com/mozilla-ai/langchain-any-llm), carrying fixes for a few issues that weren't merged upstream.

## Features

- **Unified Interface**: Use OpenAI, Anthropic, Google, or local models through a single API
- **Streaming Support**: Full support for both synchronous and asynchronous streaming
- **Tool Calling**: Native support for LangChain tool binding

## Requirements

- Python 3.11, 3.12, or 3.13

## Installation

### From this fork

This fork is not published to PyPI, so install it from git:

```bash
pip install git+https://github.com/ranzhh/langchain-any-llm
```
or
```bash
uv add git+https://github.com/ranzhh/langchain-any-llm
```
## Quick Start

**Note:** You need to have the appropriate API key available for your chosen provider. API keys can be passed explicitly via the `api_key` parameter, or set as environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). See the [any-llm documentation](https://mozilla-ai.github.io/any-llm/providers/) for provider-specific requirements.

### Basic Chat

```python
from langchain_anyllm import ChatAnyLLM

# Initialize with any supported model
llm = ChatAnyLLM(model="openai:gpt-4", temperature=0.7)

# Invoke for a single response
response = llm.invoke("Tell me a joke")
print(response.content)
```

### Streaming

```python
from langchain_anyllm import ChatAnyLLM

llm = ChatAnyLLM(model="openai:gpt-4")

# Stream responses
for chunk in llm.stream("Write a poem about the ocean"):
    print(chunk.content, end="", flush=True)
```

### Async Support

```python
import asyncio
from langchain_anyllm import ChatAnyLLM

async def main():
    llm = ChatAnyLLM(model="openai:gpt-4")

    # Async invoke
    response = await llm.ainvoke("What is the meaning of life?")
    print(response.content)

    # Async streaming
    async for chunk in llm.astream("Count to 10"):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

### Tool Calling

```python
from langchain_anyllm import ChatAnyLLM
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny!"

llm = ChatAnyLLM(model="openai:gpt-4")
llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("What's the weather in San Francisco?")
print(response.tool_calls)
```

### Configuration

```python
from langchain_anyllm import ChatAnyLLM

# Using model string with provider prefix
llm = ChatAnyLLM(
    model="openai:gpt-4",
    api_key="your-api-key",  # Optional, reads from environment if not provided
    api_base="https://custom-endpoint.com/v1",  # Optional custom endpoint
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
)

# Or using separate provider parameter
llm = ChatAnyLLM(
    model="gpt-4",
    provider="openai",
    temperature=0.7,
)

# Enable JSON mode
llm = ChatAnyLLM(
    model="openai:gpt-4",
    response_format={"type": "json_object"},
)
```

## Parameters

- `model` (str): The model to use. Can include provider prefix (e.g., "openai:gpt-4") or be used with separate `provider` parameter
- `provider` (str, optional): Provider name (e.g., "openai", "anthropic"). If not set, extracted from model string
- `api_key` (str, optional): API key for the provider. Reads from environment if not provided
- `api_base` (str, optional): Custom API endpoint
- `temperature` (float, optional): Sampling temperature (0.0 to 2.0)
- `max_tokens` (int, optional): Maximum number of tokens to generate
- `top_p` (float, optional): Nucleus sampling parameter
- `response_format` (dict, optional): Response format specification. Use `{"type": "json_object"}` for JSON mode
- `model_kwargs` (dict, optional): Additional parameters to pass to the model

## Supported Providers

any-llm supports a wide range of providers. See the [full list here](https://mozilla-ai.github.io/any-llm/providers/).


## Development

### Clone the repo

```bash
git clone https://github.com/ranzhh/langchain-any-llm.git
cd langchain-any-llm
```

### Run Tests

```bash
uv run --group test pytest tests/
```

### Type Checking

```bash
uv run --group lint mypy langchain_anyllm/
```

### Linting

```bash
uv run --group lint ruff check langchain_anyllm/
```

## License

MIT
