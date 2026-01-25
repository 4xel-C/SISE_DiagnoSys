"""
LLM Handler Module - Mistral AI.

This module provides LLM interaction management for Mistral models,
including prompt templates, API calls, and usage/cost tracking.

Example:
    >>> from app.rag.llm import llm_handler
    >>> response = llm_handler.generate("What are the symptoms?")
    >>> print(response.content)
    >>> print(response.usage)
"""

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from mistralai import Mistral

from app.rag.llm_options import (
    MODELS,
    PROMPT_TEMPLATES,
    SYSTEM_PROMPT,
    LLMResponse,
    LLMUsage,
    PromptTemplate,
    SystemPromptTemplate,
)
from app.services import LLMUsageService

load_dotenv()

logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Custom exception for missing API key."""

    pass


# Type alias for conversation history
Message = dict[str, str]  # {"role": "user"|"assistant", "content": "..."}


class LLMHandler:
    """
    Handler for Mistral LLM interactions.

    Manages model selection, prompt templates, API calls, and usage tracking.

    Example:
        >>> handler = LLMHandler(model="mistral-small")
        >>> response = handler.generate_with_template(
        ...     PromptTemplates.TRIAGE_ASSESSMENT,
        ...     patient_info="...",
        ...     context="...",
        ...     query="..."
        ... )
        >>> print(response.content)
        >>> print(f"Cost: ${response.usage.cost_usd:.6f}")
    """

    def __init__(self, model: str = "mistral-small"):
        """
        Initialize the LLM handler.

        Args:
            model: Model key from MODELS dict.
        """
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError(
                "MISTRAL_API_KEY not found in environment variables"
            )

        self._client = None
        self.set_model(model)
        self._usage_service = LLMUsageService()

        logger.info(f"LLMHandler initialized with model: {model}")

    @property
    def client(self):
        """Lazy initialization of Mistral client."""
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)

        return self._client

    def set_model(self, model: str) -> None:
        """
        Change the active model.

        Args:
            model: Model key from MODELS dict.
        """
        if model not in MODELS:
            available = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Available: {available}")

        self.config = MODELS[model]
        self.model_name = model

        logger.info(f"Model set to: {model} ({self.config.model.value})")

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of an API call."""
        input_cost = (input_tokens / 1_000_000) * self.config.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * self.config.cost_per_1m_output
        return input_cost + output_cost

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response from Mistral.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt for context.

        Returns:
            LLMResponse with content and usage stats.
        """
        logger.debug(f"Generating response with {self.model_name}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        response = self.client.chat.complete(
            model=self.config.model.value,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        latency_ms = (time.time() - start_time) * 1000

        if not response.choices or not response.usage:
            raise ValueError("Invalid response from Mistral API")

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        usage = LLMUsage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            cost_usd=self._calculate_cost(input_tokens or 0, output_tokens or 0),
            latency_ms=latency_ms,
        )

        self._track_usage(usage)

        logger.debug(
            f"Response generated: {usage.total_tokens} tokens, "
            f"${usage.cost_usd:.6f}, {usage.latency_ms:.0f}ms"
        )

        return LLMResponse(
            content=str(response.choices[0].message.content),
            usage=usage,
            model=self.config.model.value,
            raw_response=response.model_dump()
            if hasattr(response, "model_dump")
            else None,
        )

    def chat(
        self,
        new_message: str,
        history: list[Message],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response in a conversation context with history.

        Args:
            new_message: The new user message.
            history: List of previous messages [{"role": "user"|"assistant", "content": "..."}].
            system_prompt: Optional system prompt (e.g., with user name, context).

        Returns:
            LLMResponse with content and usage stats.

        Example:
            >>> history = [
            ...     {"role": "user", "content": "Bonjour, que puis-je faire pour vous?"},
            ...     {"role": "assistant", "content": "Je ressens des douleurs."},
            ... ]
            >>> response = handler.chat("Pouvez vous préciser?", history, "Tu joues le rôle d'un assistant médical.")
            >>> history.append({"role": "user", "content": "Pouvez vous préciser?"})
            >>> history.append({"role": "assistant", "content": response.content})
        """
        logger.debug(f"Chat generation with {len(history)} messages in history")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        messages.extend(history)

        # Add new user message
        messages.append({"role": "user", "content": new_message})

        start_time = time.time()
        response = self.client.chat.complete(
            model=self.config.model.value,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        latency_ms = (time.time() - start_time) * 1000

        if not response.choices or not response.usage:
            raise ValueError("Invalid response from Mistral API")

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        usage = LLMUsage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            cost_usd=self._calculate_cost(input_tokens or 0, output_tokens or 0),
            latency_ms=latency_ms,
        )

        self._track_usage(usage)

        logger.debug(
            f"Chat response generated: {usage.total_tokens} tokens, "
            f"${usage.cost_usd:.6f}, {usage.latency_ms:.0f}ms"
        )

        return LLMResponse(
            content=str(response.choices[0].message.content),
            usage=usage,
            model=self.config.model.value,
            raw_response=response.model_dump()
            if hasattr(response, "model_dump")
            else None,
        )

    def generate_with_template(
        self,
        template: PromptTemplate = PromptTemplate.DIAGNOSTIC,
        system_prompt: SystemPromptTemplate = SystemPromptTemplate.DIAGNOSYS_ASSISTANT,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using a prompt template.

        Args:
            template: The prompt template to use (default: DIAGNOSTIC).
            system_prompt: System prompt template for context.
            **kwargs: Variables for the template: context, documents_chunks, patient_chunks, query.

        Returns:
            LLMResponse with content and usage stats.
        """
        if template not in PROMPT_TEMPLATES:
            available = ", ".join(t.name for t in PromptTemplate)
            raise ValueError(f"Unknown template '{template}'. Available: {available}")
        elif system_prompt not in SYSTEM_PROMPT:
            available = ", ".join(t.name for t in SystemPromptTemplate)
            raise ValueError(
                f"Unknown system prompt '{system_prompt}'. Available: {available}"
            )

        prompt = PROMPT_TEMPLATES[template].format(**kwargs)
        return self.generate(prompt, SYSTEM_PROMPT[system_prompt])

    def _track_usage(self, usage: LLMUsage, success: bool = True) -> None:
        """
        Track usage statistics by recording to the database.

        Args:
            usage: LLMUsage object with token counts and latency.
            success: Whether the request was successful.
        """
        self._usage_service.record_usage(
            model_name=self.model_name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            response_time_ms=usage.latency_ms,
            success=success,
        )

    def get_total_usage(self) -> dict:
        """Get cumulative usage statistics from database."""
        return self._usage_service.get_summary()

    def get_today_stats(self) -> dict:
        """Get today's usage statistics."""
        return {
            "total_tokens": self._usage_service.get_total_tokens_today(),
            "total_requests": self._usage_service.get_total_requests_today(),
            "current_model": self.model_name,
            "model_id": self.config.model.value,
        }

    @staticmethod
    def list_models() -> list[str]:
        """List available model keys."""
        return list(MODELS.keys())


# Default singleton instance
llm_handler = LLMHandler(model=os.getenv("LLM_MODEL", "mistral-small"))
