import logging
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Any

import asyncio
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

RETRY_MAX_SECONDS = 120
RETRY_BASE_DELAY = 1.0
PER_REQUEST_TIMEOUT = 1800  # Hard per-API-call timeout via multiprocessing (30 min for long-thinking models)

if TYPE_CHECKING:
    from transformers import Pipeline


def _call_api_in_subprocess(result_queue: multiprocessing.Queue,
                             api_key: str, base_url: str, kwargs: dict) -> None:
    """Run a single chat.completions.create() call in a subprocess.

    Puts (text, usage_dict) or (None, None) into result_queue.
    Uses a separate process to guarantee that hung TCP connections can be killed.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=1800.0, max_retries=0)
        resp = client.chat.completions.create(**kwargs)
        usage = {"prompt_tokens": resp.usage.prompt_tokens,
                 "completion_tokens": resp.usage.completion_tokens,
                 "total_tokens": resp.usage.total_tokens}
        result_queue.put((resp.choices[0].message.content, usage))
    except Exception:
        result_queue.put((None, None))


class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> tuple[Optional[str], Optional[dict]]:
        """
        Sends the prompt to the LLM and returns the response as a string.
        """
        pass

    @abstractmethod
    def complete_batch(self, prompts: list[str]) -> list[str]:
        """
        Sends a list of prompts to the LLM and returns a list of responses.
        Clients that support true batch execution should override this, otherwise throw an exception.
        """
        pass


class OpenAIClient(BaseLLMClient):
    def __init__(self, model_id: str = "o3-mini", concurrency: int = 30, base_url: str | None = None, temperature: float | None = None, top_p: float | None = None, top_k: int | None = None, min_p: float | None = None, presence_penalty: float | None = None, repetition_penalty: float | None = None, reasoning_effort: str | None = None, thinking: bool = False, max_tokens: int | None = None, timeout: float = 1800.0) -> None:
        """
        Initialize the OpenAI LLMClient with an API key from environment variables.

        :param reasoning_effort: Reasoning effort level for thinking mode (\"high\" or \"max\"). None disables.
        :param thinking: Whether to enable thinking mode via chat_template_kwargs.
        :param max_tokens: Maximum tokens for completion. None uses the model default.
        :param timeout: HTTP timeout in seconds (default: 1800 = 30 min).
        """
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=3)
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        logging.info(f"Loaded OpenAI model: {model_id}")

        self.model_id = model_id
        self.concurrency = concurrency
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.max_tokens = max_tokens

    def _build_kwargs(self, messages: list[dict]) -> dict:
        """Build kwargs dict for chat completions, routing non-standard params through extra_body."""
        kwargs: dict = {"model": self.model_id, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        extra_body: dict = {}
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if self.thinking:
            extra_body["chat_template_kwargs"] = {"thinking": True}
        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs


    def complete(self, prompt: str) -> tuple[Optional[str], Optional[dict]]:
        messages = [{"role": "user", "content": prompt}]
        kwargs = self._build_kwargs(messages)
        response = self.client.chat.completions.create(**kwargs)

        usage = response.usage
        text = response.choices[0].message.content
        return text, usage

    def _call_with_retry(self, prompt: str) -> tuple[Optional[str], Optional[dict]]:
        """Call the LLM with retry logic. Each API call runs in a subprocess
        with a hard timeout to guarantee termination of hung TCP connections."""
        deadline = time.monotonic() + RETRY_MAX_SECONDS
        delay = RETRY_BASE_DELAY
        kwargs = self._build_kwargs([{"role": "user", "content": prompt}])
        while True:
            try:
                ctx = multiprocessing.get_context("spawn")
                queue: multiprocessing.Queue = ctx.Queue()
                proc = ctx.Process(
                    target=_call_api_in_subprocess,
                    args=(queue, self.api_key, self.base_url, kwargs),
                )
                proc.start()
                proc.join(timeout=PER_REQUEST_TIMEOUT)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    raise TimeoutError(f"API call timed out after {PER_REQUEST_TIMEOUT}s")
                if not queue.empty():
                    text, usage = queue.get()
                    return text, usage
                raise RuntimeError(f"API subprocess returned no result (exit code {proc.exitcode})")
            except Exception as e:
                msg = str(e)[:120]
                if time.monotonic() >= deadline:
                    logging.error("LLM call failed after %.0fs retries: %s", RETRY_MAX_SECONDS, msg)
                    return None, None
                logging.warning("LLM call failed, retrying in %.1fs: %s", delay, msg)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    def complete_batch(self, prompts: list[str]) -> list[tuple[Optional[str], Optional[dict]]]:
        """Run batch of prompts concurrently using thread pool."""
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(self._call_with_retry, p): i for i, p in enumerate(prompts)}
            results = [None] * len(prompts)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    # Each _call_with_retry has its own 120s deadline,
                    # so 240s outer timeout is a generous safety net.
                    results[idx] = future.result(timeout=RETRY_MAX_SECONDS + 120)
                except Exception as e:
                    logging.error("Thread for prompt %d failed: %s", idx, e)
                    results[idx] = (None, None)
        return results


class HuggingFaceClient(BaseLLMClient):
    def __init__(self, max_new_tokens: int, batch_size: int, model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", temperature: float | None = None, top_p: float | None = None, top_k: int | None = None, min_p: float | None = None, presence_penalty: float | None = None, repetition_penalty: float | None = None) -> None:
        """
        Load the Hugging Face model during initialization.
        Make sure that the user is authenticated into huggingface hub.

        :param max_new_tokens: The maximum number of new tokens to generate.
        :param batch_size: The batch size for processing, suggested maximum of 4, depending on the gpu.
        :param model_id: The model ID to load from Hugging Face hub.
        """
        import torch
        from transformers import pipeline

        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        logging.info(f"HuggingFaceClient initialized with max_new_tokens={max_new_tokens}, batch_size={batch_size}, model_id={model_id}")

        # Load the model from Hugging Face hub
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            logging.info(f"Loaded Hugging Face model: {model_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_id} model from huggingface hub: {e}."
                               f"Make sure you are authenticated in to the huggingface hub.")

    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        logging.debug("Sending prompt to Hugging Face model: %s", messages)

        pipeline_kwargs = {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature or 0.6}
        if self.top_p is not None:
            pipeline_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            pipeline_kwargs["top_k"] = self.top_k
        if self.min_p is not None:
            pipeline_kwargs["min_p"] = self.min_p
        if self.presence_penalty is not None:
            pipeline_kwargs["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty is not None:
            pipeline_kwargs["repetition_penalty"] = self.repetition_penalty
        outputs = self.pipeline(messages, **pipeline_kwargs)
        logging.debug("Raw outputs from sequential pipeline: %s", outputs)

        try:
            return outputs[0]['generated_text'][-1]["content"]
        except Exception as e:
            raise RuntimeError(f"Error extracting generated text: {e}")

    def complete_batch(self, prompts: list[str]) -> list[str]:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        logging.debug("Sending batch of prompts to Hugging Face model: %s", messages)

        pipeline_kwargs = {"max_new_tokens": self.max_new_tokens, "batch_size": self.batch_size, "temperature": self.temperature or 0.6}
        if self.top_p is not None:
            pipeline_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            pipeline_kwargs["top_k"] = self.top_k
        if self.min_p is not None:
            pipeline_kwargs["min_p"] = self.min_p
        if self.presence_penalty is not None:
            pipeline_kwargs["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty is not None:
            pipeline_kwargs["repetition_penalty"] = self.repetition_penalty
        outputs = self.pipeline(messages, **pipeline_kwargs)
        logging.debug("Raw outputs from batch pipeline: %s", outputs)

        try:
            return [output[0]['generated_text'][-1]["content"] for output in outputs]
        except Exception as e:
            raise RuntimeError(f"Error extracting generated text in batch: {e}")


class DeepSeekClient(BaseLLMClient):
    """
    Sync DeepSeek client using OpenAI client with thread-pool for batch calls.
    Uses OS-level TCP timeouts for reliable connection handling.
    """
    def __init__(self, concurrency: int = 30, model_id: str = "deepseek-reasoner", base_url: str = "https://api.deepseek.com", temperature: float | None = None, top_p: float | None = None, top_k: int | None = None, min_p: float | None = None, presence_penalty: float | None = None, repetition_penalty: float | None = None):
        load_dotenv()
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.concurrency = concurrency
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty

        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=3)

    def _build_kwargs(self, messages: list[dict]) -> dict:
        """Build kwargs dict for chat completions, routing non-standard params through extra_body."""
        kwargs: dict = {"model": self.model_id, "messages": messages, "stream": False, "max_tokens": 9000}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty

        extra_body: dict = {}
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs

    def complete(self, prompt: str) -> tuple[str, Any]:
        """Single-prompt call using the sync client."""
        kwargs = self._build_kwargs([{"role": "user", "content": prompt}])
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content, resp.usage

    def _call_with_retry(self, prompt: str) -> tuple[Optional[str], Optional[dict]]:
        """Call the LLM with retry logic. Each API call runs in a subprocess
        with a hard timeout to guarantee termination of hung TCP connections."""
        deadline = time.monotonic() + RETRY_MAX_SECONDS
        delay = RETRY_BASE_DELAY
        kwargs = self._build_kwargs([{"role": "user", "content": prompt}])
        while True:
            try:
                ctx = multiprocessing.get_context("spawn")
                queue: multiprocessing.Queue = ctx.Queue()
                proc = ctx.Process(
                    target=_call_api_in_subprocess,
                    args=(queue, self.api_key, self.base_url, kwargs),
                )
                proc.start()
                proc.join(timeout=PER_REQUEST_TIMEOUT)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    raise TimeoutError(f"API call timed out after {PER_REQUEST_TIMEOUT}s")
                if not queue.empty():
                    text, usage = queue.get()
                    return text, usage
                raise RuntimeError(f"API subprocess returned no result (exit code {proc.exitcode})")
            except Exception as e:
                msg = str(e)[:120]
                if time.monotonic() >= deadline:
                    logging.error("LLM call failed after %.0fs retries: %s", RETRY_MAX_SECONDS, msg)
                    return None, None
                logging.warning("LLM call failed, retrying in %.1fs: %s", delay, msg)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    def complete_batch(self, prompts: list[str]) -> list[tuple[str, Any]]:
        """Batch call using thread pool with the sync client."""
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(self._call_with_retry, p): i for i, p in enumerate(prompts)}
            results = [None] * len(prompts)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=RETRY_MAX_SECONDS + 120)
                except Exception as e:
                    logging.error("Thread for prompt %d failed: %s", idx, e)
                    results[idx] = (None, None)
        return results
