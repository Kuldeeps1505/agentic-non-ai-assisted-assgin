
"""
Assignment 01 — Raw LLM Engineering

Covers:
- HTTP / REST
- Authentication
- Messages / roles
- Generation parameters
- Streaming / SSE
- TTFT
- Timeouts
- Retries with exponential backoff
- Error handling
- Logging
- Latency measurement
- Environment variables
- Provider abstraction
- OpenAI-compatible providers
- Ollama support

Run:
    python llm_client.py

Environment:
    OPENAI_API_KEY=...

Optional:
    OPENAI_MODEL=...
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=...

Dependencies:
    pip install requests
"""

import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Generator

import requests


# ============================================================
# Configuration
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# Data Models
# ============================================================

@dataclass
class LLMResponse:
    """
    Normalized response returned by every provider.
    """

    content: str
    provider: str
    model: str

    latency: float
    time_to_first_token: Optional[float]

    usage: Optional[Dict] = None


# ============================================================
# Exceptions
# ============================================================

class LLMError(Exception):
    """Base LLM client error."""


class AuthenticationError(LLMError):
    """Authentication / API key error."""


class RateLimitError(LLMError):
    """Rate limit error."""


class ProviderError(LLMError):
    """Provider-side error."""


class InvalidResponseError(LLMError):
    """Unexpected provider response."""


# ============================================================
# HTTP Client
# ============================================================

class HTTPClient:
    """
    Small HTTP wrapper responsible for:

    - requests
    - timeouts
    - retries
    - exponential backoff
    - HTTP error handling
    """

    def __init__(
        self,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES
    ):
        self.timeout = timeout
        self.max_retries = max_retries

    def post(
        self,
        url: str,
        headers: Dict,
        payload: Dict,
        stream: bool = False
    ) -> requests.Response:

        for attempt in range(self.max_retries + 1):

            try:

                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=stream
                )

                # Success
                if 200 <= response.status_code < 300:
                    return response

                # Authentication
                if response.status_code in (401, 403):
                    raise AuthenticationError(
                        self._error_message(response)
                    )

                # Rate limiting
                if response.status_code == 429:
                    if attempt < self.max_retries:
                        self._backoff(attempt)
                        continue

                    raise RateLimitError(
                        self._error_message(response)
                    )

                # Retryable server errors
                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        self._backoff(attempt)
                        continue

                    raise ProviderError(
                        self._error_message(response)
                    )

                # Other HTTP errors
                raise ProviderError(
                    self._error_message(response)
                )

            except requests.exceptions.Timeout:

                if attempt < self.max_retries:

                    logger.warning(
                        "Request timed out. Retrying..."
                    )

                    self._backoff(attempt)

                    continue

                raise LLMError(
                    "Request timed out after retries."
                )

            except requests.exceptions.ConnectionError:

                if attempt < self.max_retries:

                    logger.warning(
                        "Connection failed. Retrying..."
                    )

                    self._backoff(attempt)

                    continue

                raise LLMError(
                    "Could not connect to provider."
                )

            except requests.exceptions.RequestException as exc:

                raise LLMError(
                    f"HTTP request failed: {exc}"
                )

        raise LLMError("Request failed unexpectedly.")

    def _backoff(self, attempt: int):

        delay = INITIAL_BACKOFF * (2 ** attempt)

        logger.info(
            "Retrying in %.1f seconds...",
            delay
        )

        time.sleep(delay)

    @staticmethod
    def _error_message(
        response: requests.Response
    ) -> str:

        try:
            data = response.json()

            return (
                f"HTTP {response.status_code}: "
                f"{data}"
            )

        except ValueError:

            return (
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )


# ============================================================
# Provider Interface
# ============================================================

class LLMProvider(ABC):
    """
    Common interface.

    Application code talks to this interface
    instead of directly depending on a provider.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> LLMResponse:
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> Generator[str, None, None]:
        pass


# ============================================================
# OpenAI Provider
# ============================================================

class OpenAIProvider(LLMProvider):

    ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str,
        http_client: HTTPClient
    ):

        if not api_key:
            raise AuthenticationError(
                "OPENAI_API_KEY is not set."
            )

        self.api_key = api_key
        self.model = model
        self.http = http_client

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _build_payload(
        self,
        messages,
        temperature,
        max_tokens,
        stream=False
    ):

        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

    def generate(
        self,
        messages,
        temperature=0.7,
        max_tokens=200
    ) -> LLMResponse:

        start = time.perf_counter()

        payload = self._build_payload(
            messages,
            temperature,
            max_tokens,
            stream=False
        )

        response = self.http.post(
            self.ENDPOINT,
            self.headers,
            payload,
            stream=False
        )

        latency = time.perf_counter() - start

        try:

            data = response.json()

            content = data["choices"][0]["message"]["content"]

            usage = data.get("usage")

        except (ValueError, KeyError, IndexError) as exc:

            raise InvalidResponseError(
                f"Invalid provider response: {exc}"
            )

        return LLMResponse(
            content=content,
            provider="OpenAI",
            model=self.model,
            latency=latency,
            time_to_first_token=None,
            usage=usage
        )

    def stream(
        self,
        messages,
        temperature=0.7,
        max_tokens=200
    ):

        payload = self._build_payload(
            messages,
            temperature,
            max_tokens,
            stream=True
        )

        response = self.http.post(
            self.ENDPOINT,
            self.headers,
            payload,
            stream=True
        )

        start = time.perf_counter()

        first_token_time = None

        try:

            for line in response.iter_lines(
                decode_unicode=True
            ):

                if not line:
                    continue

                # SSE format:
                # data: {...}

                if not line.startswith("data:"):
                    continue

                data_string = line[5:].strip()

                if data_string == "[DONE]":
                    break

                try:

                    data = json.loads(data_string)

                except json.JSONDecodeError:
                    logger.warning(
                        "Could not parse SSE event."
                    )
                    continue

                try:

                    content = (
                        data["choices"][0]
                        ["delta"]
                        .get("content")
                    )

                except (KeyError, IndexError):

                    continue

                if content:

                    if first_token_time is None:

                        first_token_time = (
                            time.perf_counter() - start
                        )

                        logger.info(
                            "TTFT: %.3f seconds",
                            first_token_time
                        )

                    yield content

        finally:

            response.close()


# ============================================================
# Ollama Provider
# ============================================================

class OllamaProvider(LLMProvider):

    def __init__(
        self,
        base_url: str,
        model: str,
        http_client: HTTPClient
    ):

        self.endpoint = (
            f"{base_url.rstrip('/')}"
            "/api/chat"
        )

        self.model = model
        self.http = http_client

        self.headers = {
            "Content-Type": "application/json"
        }

    def generate(
        self,
        messages,
        temperature=0.7,
        max_tokens=200
    ):

        start = time.perf_counter()

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        response = self.http.post(
            self.endpoint,
            self.headers,
            payload,
            stream=False
        )

        latency = time.perf_counter() - start

        try:

            data = response.json()

            content = data["message"]["content"]

        except (ValueError, KeyError) as exc:

            raise InvalidResponseError(
                f"Invalid Ollama response: {exc}"
            )

        return LLMResponse(
            content=content,
            provider="Ollama",
            model=self.model,
            latency=latency,
            time_to_first_token=None,
            usage=None
        )

    def stream(
        self,
        messages,
        temperature=0.7,
        max_tokens=200
    ):

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }

        response = self.http.post(
            self.endpoint,
            self.headers,
            payload,
            stream=True
        )

        start = time.perf_counter()

        first_token_time = None

        try:

            for line in response.iter_lines(
                decode_unicode=True
            ):

                if not line:
                    continue

                try:

                    data = json.loads(line)

                except json.JSONDecodeError:
                    continue

                content = (
                    data.get("message", {})
                    .get("content")
                )

                if content:

                    if first_token_time is None:

                        first_token_time = (
                            time.perf_counter() - start
                        )

                        logger.info(
                            "TTFT: %.3f seconds",
                            first_token_time
                        )

                    yield content

                if data.get("done"):
                    break

        finally:

            response.close()


# ============================================================
# CLI Application
# ============================================================

class CLI:

    def __init__(self, provider: LLMProvider):

        self.provider = provider

        self.messages = []

    def run(self):

        print("\nRaw LLM CLI")
        print("Commands:")
        print("  /exit    → quit")
        print("  /clear   → clear conversation")
        print("  /stream  → toggle streaming")
        print()

        streaming = True

        while True:

            try:

                user_prompt = input("You: ").strip()

            except (KeyboardInterrupt, EOFError):

                print("\nGoodbye.")
                break

            if not user_prompt:
                continue

            if user_prompt.lower() == "/exit":

                break

            if user_prompt.lower() == "/clear":

                self.messages.clear()

                print("Conversation cleared.")

                continue

            if user_prompt.lower() == "/stream":

                streaming = not streaming

                print(
                    f"Streaming: {streaming}"
                )

                continue

            self.messages.append(
                {
                    "role": "user",
                    "content": user_prompt
                }
            )

            try:

                if streaming:

                    self._stream_response()

                else:

                    self._normal_response()

            except LLMError as exc:

                logger.error("%s", exc)

    def _normal_response(self):

        response = self.provider.generate(
            self.messages
        )

        print(
            f"Assistant: {response.content}"
        )

        print(
            f"[Latency: {response.latency:.3f}s]"
        )

        self.messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )

    def _stream_response(self):

        print("Assistant: ", end="", flush=True)

        start = time.perf_counter()

        first_token_time = None

        full_response = []

        for chunk in self.provider.stream(
            self.messages
        ):

            if first_token_time is None:

                first_token_time = (
                    time.perf_counter() - start
                )

            print(
                chunk,
                end="",
                flush=True
            )

            full_response.append(chunk)

        total_latency = (
            time.perf_counter() - start
        )

        content = "".join(full_response)

        print()

        print(
            f"[TTFT: {first_token_time:.3f}s]"
            if first_token_time is not None
            else "[TTFT: unavailable]"
        )

        print(
            f"[Total latency: {total_latency:.3f}s]"
        )

        self.messages.append(
            {
                "role": "assistant",
                "content": content
            }
        )


# ============================================================
# Provider Factory
# ============================================================

def create_provider(
    provider_name: str,
    http_client: HTTPClient
) -> LLMProvider:

    provider_name = provider_name.lower()

    if provider_name == "openai":

        return OpenAIProvider(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            http_client=http_client
        )

    if provider_name == "ollama":

        return OllamaProvider(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            http_client=http_client
        )

    raise ValueError(
        f"Unsupported provider: {provider_name}"
    )


# ============================================================
# Main
# ============================================================

def main():

    provider_name = os.getenv(
        "LLM_PROVIDER",
        "openai"
    )

    http_client = HTTPClient()

    try:

        provider = create_provider(
            provider_name,
            http_client
        )

        logger.info(
            "Using provider: %s",
            provider_name
        )

        cli = CLI(provider)

        cli.run()

    except LLMError as exc:

        logger.error(
            "Startup failed: %s",
            exc
        )

        sys.exit(1)

    except ValueError as exc:

        logger.error(
            "Configuration error: %s",
            exc
        )

        sys.exit(1)


if __name__ == "__main__":
    main()