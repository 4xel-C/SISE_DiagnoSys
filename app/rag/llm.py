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
from ecologits import EcoLogits
from mistralai import Mistral

from app.rag.llm_options import (
    MODELS,
    PROMPT_TEMPLATES,
    SYSTEM_PROMPT,
    LLMResponse,
    LLMUsage,
    MistralModel,
    PromptTemplate,
    SystemPromptTemplate,
)

load_dotenv()

# Initialize EcoLogits for ennvironmental impact tracking
EcoLogits.init(providers=["mistralai"], electricity_mix_zone="FRA")

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
        >>> handler = LLMHandler(model="mistral-small-latest")
        >>> response = handler.generate_with_template(
        ...     PromptTemplates.TRIAGE_ASSESSMENT,
        ...     patient_info="...",
        ...     context="...",
        ...     query="..."
        ... )
        >>> print(response.content)
        >>> print(f"Cost: ${response.usage.cost_usd:.6f}")
    """

    def __init__(self, model: MistralModel = MistralModel.MISTRAL_SMALL):
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

        logger.info(f"LLMHandler initialized with model: {model.value}")

    @property
    def client(self):
        """Lazy initialization of Mistral client."""
        if self._client is None:
            self._client = Mistral(api_key=self.api_key)

        return self._client

    def set_model(self, model: MistralModel) -> None:
        """
        Change the active model.

        Args:
            model: Model key from MODELS dict.
        """
        self.config = MODELS[model.value]
        self.model_name = model.value

        logger.info(f"Model set to: {model.value} ({self.config.model.value})")

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

        usage = self._generate_usages(response, latency_ms)

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

        # Call the private method to compute usages
        usage = self._generate_usages(response, latency_ms)

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

    def _generate_usages(self, response, latency: float) -> LLMUsage:
        """Private methode to generate usage with ecological impact estimates."""

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = self._calculate_cost(input_tokens or 0, output_tokens or 0)
        latency = latency

        # Get all ecological impact values, defaulting to 0 if not present from Ecologits
        energy = (
            response.impacts.energy.value
            if response.impacts and response.impacts.energy
            else 0
        )
        gwp = (
            response.impacts.gwp.value
            if response.impacts and response.impacts.gwp
            else 0
        )
        adpe = (
            response.impacts.adpe.value
            if response.impacts and response.impacts.adpe
            else 0
        )
        pe = (
            response.impacts.pe.value if response.impacts and response.impacts.pe else 0
        )
        wcf = (
            response.impacts.wcf.value
            if response.impacts and response.impacts.wcf
            else 0
        )

        usages = LLMUsage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            cost_usd=cost,
            latency_ms=latency,
            energy_kwh=energy,
            gwp_kgCO2eq=gwp,
            adpe_mgSbEq=adpe,
            pd_mj=pe,
            wcf_liters=wcf,
        )

        return usages

    @staticmethod
    def list_models() -> list[str]:
        """List available model keys."""
        return list(MODELS.keys())


# Default instance
llm_handler = LLMHandler(
    model=MistralModel(os.getenv("LLM_MODEL", "mistral-small-latest"))
)

# Instance for contextupdate
llm_context_updator = LLMHandler(model=MistralModel("ministral-3b-latest"))
