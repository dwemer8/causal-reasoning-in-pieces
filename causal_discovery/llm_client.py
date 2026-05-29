import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Any

import asyncio
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

RETRY_MAX_SECONDS = 120
RETRY_BASE_DELAY = 1.0

if TYPE_CHECKING:
    from transformers import Pipeline


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
    def __init__(self, model_id: str = "o3-mini", concurrency: int = 30, base_url: str | None = None, temperature: float | None = None, top_p: float | None = None, top_k: int | None = None, min_p: float | None = None, presence_penalty: float | None = None, repetition_penalty: float | None = None) -> None:
        """
        Initialize the OpenAI LLMClient with an API key from environment variables.
        """
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=3)
        self.api_key = api_key
        self.base_url = base_url
        logging.info(f"Loaded OpenAI model: {model_id}")

        self.model_id = model_id
        self.concurrency = concurrency
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty

    def _build_kwargs(self, messages: list[dict]) -> dict:
        """Build kwargs dict for chat completions, routing non-standard params through extra_body."""
        kwargs: dict = {"model": self.model_id, "messages": messages}
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


    def complete(self, prompt: str) -> tuple[Optional[str], Optional[dict]]:
        messages = [{"role": "user", "content": prompt}]
        kwargs = self._build_kwargs(messages)
        response = self.client.chat.completions.create(**kwargs)

        usage = response.usage
        text = response.choices[0].message.content
        return text, usage

    def complete_batch(self, prompts: list[str]) -> list[str]:
        return asyncio.run(self._complete_batch_async(prompts))

    async def _complete_batch_async(self, prompts: list[str]) -> list[tuple[Optional[str], Optional[dict]]]:
        async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120.0, max_retries=3)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _call(p: str) -> tuple[Optional[str], Optional[dict]]:
            deadline = time.monotonic() + RETRY_MAX_SECONDS
            delay = RETRY_BASE_DELAY
            while True:
                try:
                    async with semaphore:
                        kwargs = self._build_kwargs([{"role": "user", "content": p}])
                        resp = await async_client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content, resp.usage
                except Exception as e:
                    if time.monotonic() >= deadline:
                        logging.error("LLM call failed after %.0fs retries: %s", RETRY_MAX_SECONDS, e)
                        return None, None
                    logging.warning("LLM call failed, retrying in %.1fs: %s", delay, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

        try:
            tasks = []
            for i, p in enumerate(prompts):
                if i > 0:
                    await asyncio.sleep(0.3)
                tasks.append(asyncio.create_task(_call(p)))

            return await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=RETRY_MAX_SECONDS + 60,
            )
        except asyncio.TimeoutError:
            logging.error("Batch call timed out after %.0fs.", RETRY_MAX_SECONDS + 60)
            return [(None, None)] * len(prompts)
        finally:
            await async_client.close()


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
    Async DeepSeek client using AsyncOpenAI under the hood but exposes
    the sync interface for compatibility with a pipeline.
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

    def _build_kwargs(self, messages: list[dict]) -> dict:
        """Build kwargs dict for chat completions, routing non-standard params through extra_body."""
        kwargs: dict = {"model": self.model_id, "messages": messages, "stream": False}
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
        """
        Single-prompt call: wraps the async call in asyncio.run
        Returns (response_text, usage)
        """
        return asyncio.run(self._complete_async(prompt))

    async def _complete_async(self, prompt: str) -> tuple[str, Any]:
        async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            kwargs = self._build_kwargs([{"role": "user", "content": prompt}])
            resp = await async_client.chat.completions.create(**kwargs)
            usage = resp.usage
            text = resp.choices[0].message.content
            return text, usage
        finally:
            await async_client.close()

    def complete_batch(self, prompts: list[str]) -> list[tuple[str, Any]]:
        """
        Batch call: runs all prompts concurrently within a single event loop
        Returns a list of (response_text, usage) tuples
        """
        return asyncio.run(self._complete_batch_async(prompts))

    async def _complete_batch_async(self, prompts: list[str]) -> list[tuple[str, Any]]:
        async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _call(p: str) -> tuple[str, Any]:
            deadline = time.monotonic() + RETRY_MAX_SECONDS
            delay = RETRY_BASE_DELAY
            while True:
                try:
                    async with semaphore:
                        kwargs = self._build_kwargs([{"role": "user", "content": p}])
                        resp = await async_client.chat.completions.create(**kwargs)
                    text = resp.choices[0].message.content
                    usage = resp.usage
                    return text, usage
                except Exception as e:
                    if time.monotonic() >= deadline:
                        logging.error("LLM call failed after %.0fs retries: %s", RETRY_MAX_SECONDS, e)
                        return None, None
                    logging.warning("LLM call failed, retrying in %.1fs: %s", delay, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

        try:
            tasks = []
            for i, p in enumerate(prompts):
                if i > 0:
                    await asyncio.sleep(0.3)
                tasks.append(asyncio.create_task(_call(p)))
            return await asyncio.gather(*tasks)
        finally:
            await async_client.close()
