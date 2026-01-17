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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Custom exception for missing API key."""

    pass


class MistralModel(Enum):
    """Available Mistral models."""

    MISTRAL_SMALL = "mistral-small-latest"
    MISTRAL_MEDIUM = "mistral-medium-latest"
    MISTRAL_LARGE = "mistral-large-latest"
    CODESTRAL = "codestral-latest"
    MINISTRAL_8B = "ministral-8b-latest"
    MINISTRAL_3B = "ministral-3b-latest"


@dataclass
class ModelConfig:
    """Configuration for a Mistral model."""

    model: MistralModel
    temperature: float = 0.7
    max_tokens: int = 1024
    cost_per_1m_input: float = 0.0  # Cost in USD per 1M input tokens
    cost_per_1m_output: float = 0.0  # Cost in USD per 1M output tokens


@dataclass
class LLMUsage:
    """Token usage and cost tracking for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    usage: LLMUsage
    model: str
    raw_response: Optional[dict] = None


# TODO: check and update costs
# Pre-configured Mistral models with pricing (USD per 1M tokens)
MODELS: dict[str, ModelConfig] = {
    "ministral-3b": ModelConfig(
        model=MistralModel.MINISTRAL_3B,
        cost_per_1m_input=0.04,
        cost_per_1m_output=0.04,
    ),
    "ministral-8b": ModelConfig(
        model=MistralModel.MINISTRAL_8B,
        cost_per_1m_input=0.1,
        cost_per_1m_output=0.1,
    ),
    "mistral-small": ModelConfig(
        model=MistralModel.MISTRAL_SMALL,
        cost_per_1m_input=0.2,
        cost_per_1m_output=0.6,
    ),
    "mistral-medium": ModelConfig(
        model=MistralModel.MISTRAL_MEDIUM,
        cost_per_1m_input=2.5,
        cost_per_1m_output=7.5,
    ),
    "mistral-large": ModelConfig(
        model=MistralModel.MISTRAL_LARGE,
        cost_per_1m_input=2.0,
        cost_per_1m_output=6.0,
    ),
    "codestral": ModelConfig(
        model=MistralModel.CODESTRAL,
        cost_per_1m_input=0.3,
        cost_per_1m_output=0.9,
    ),
}

TEMPLATE = """
    Tu es un assistant médical nommé DiagnoSys, spécialisé dans l'aide au diagnostic clinique basé sur les informations fournies par le 
    patient et le médecine par transcription audio et le contexte médical pertinent.

    Contexte medical du patient:
    {context}

    Documents médicaux pertinents:
    {documents_chunks}

    Patients similaires:
    {patient_chunks}

    Conversation avec le patient:
    {query}

    Mets à jour le contexte médical du patient en fonction des nouvelles informations fournies dans la conversation. 
    Propose succinctement aux maximimums 3 diagnostics différents et expose ton rationnel.
    Réponse en français:
    """


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
        self._total_usage = LLMUsage()
        self._call_history: list[LLMUsage] = []

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

    def generate_with_template(
        self,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using a prompt template.

        Args:
            system_prompt: Optional system prompt for context.
            **kwargs: Variables for the template: context, documents_chunks, query.

        Returns:
            LLMResponse with content and usage stats.
        """
        prompt = TEMPLATE.format(**kwargs)
        return self.generate(prompt, system_prompt)

    def _track_usage(self, usage: LLMUsage) -> None:
        """Track cumulative usage statistics."""
        self._call_history.append(usage)
        self._total_usage.input_tokens += usage.input_tokens
        self._total_usage.output_tokens += usage.output_tokens
        self._total_usage.total_tokens += usage.total_tokens
        self._total_usage.cost_usd += usage.cost_usd

    def get_total_usage(self) -> LLMUsage:
        """Get cumulative usage statistics."""
        return self._total_usage

    def get_session_stats(self) -> dict:
        """Get detailed session statistics."""
        return {
            "total_calls": len(self._call_history),
            "total_input_tokens": self._total_usage.input_tokens,
            "total_output_tokens": self._total_usage.output_tokens,
            "total_tokens": self._total_usage.total_tokens,
            "total_cost_usd": self._total_usage.cost_usd,
            "current_model": self.model_name,
            "model_id": self.config.model.value,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._total_usage = LLMUsage()
        self._call_history.clear()
        logger.info("Usage statistics reset")

    @staticmethod
    def list_models() -> list[str]:
        """List available model keys."""
        return list(MODELS.keys())


# Default singleton instance
llm_handler = LLMHandler(model=os.getenv("LLM_MODEL", "mistral-small"))
