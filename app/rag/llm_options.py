from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


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
    total_tokens: int = 0  # calculated in __post_init__
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    energy_kwh: Optional[float] = None
    gwp_kgCO2eq: Optional[float] = None
    adpe_mgSbEq: Optional[float] = None
    pd_mj: Optional[float] = None
    wcf_liters: Optional[float] = None

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


# Pre-configured Mistral models with pricing (USD per 1M tokens)
MODELS: dict[str, ModelConfig] = {
    "ministral-3b-latest": ModelConfig(
        model=MistralModel.MINISTRAL_3B,
        cost_per_1m_input=0.04,
        cost_per_1m_output=0.04,
    ),
    "ministral-8b-latest": ModelConfig(
        model=MistralModel.MINISTRAL_8B,
        cost_per_1m_input=0.1,
        cost_per_1m_output=0.1,
    ),
    "mistral-small-latest": ModelConfig(
        model=MistralModel.MISTRAL_SMALL,
        cost_per_1m_input=0.2,
        cost_per_1m_output=0.6,
    ),
    "mistral-medium-latest": ModelConfig(
        model=MistralModel.MISTRAL_MEDIUM,
        cost_per_1m_input=2.5,
        cost_per_1m_output=7.5,
    ),
    "mistral-large-latest": ModelConfig(
        model=MistralModel.MISTRAL_LARGE,
        cost_per_1m_input=2.0,
        cost_per_1m_output=6.0,
    ),
    "codestral-latest": ModelConfig(
        model=MistralModel.CODESTRAL,
        cost_per_1m_input=0.3,
        cost_per_1m_output=0.9,
    ),
}


class PromptTemplate(Enum):
    """Available prompt templates."""

    DIAGNOSTIC = "diagnostic"
    SUMMARY = "summary"
    CONVERSATION = "conversation"


class SystemPromptTemplate(Enum):
    """Available system prompt templates"""

    DIAGNOSYS_ASSISTANT = "diagnosys_assistant"
    CONTEXT_UPDATER = "context_updater"
    CONVERSATION = "conversation"


# ===================================================  PROMPTS  ===================================================
# Template for each prompt
PROMPT_TEMPLATES: dict[PromptTemplate, str] = {
    PromptTemplate.DIAGNOSTIC: """
Contexte médical du patient:
{context}

Documents médicaux pertinents:
{documents_chunks}

Diagnostiques:
""",
    PromptTemplate.SUMMARY: """
Contexte médical du patient:
{context}

Conversation avec le patient:
{audio}

Résume de manière concise les informations clés du patient et les points importants à retenir.
Réponse en français:
""",
    PromptTemplate.CONVERSATION: """
    historique de la conversation:
{conversation_history}
    Nouveau message du patient:
{new_message}
""",
}


# =================================================== SYSTEM PROMPTS ===================================================
SYSTEM_PROMPT: dict[SystemPromptTemplate, str] = {
    SystemPromptTemplate.DIAGNOSYS_ASSISTANT: """
Tu es un assistant médical nommé DiagnoSys, spécialisé dans l'aide au diagnostic clinique basé sur les informations fournies par le
médecin. Tu disposes du contexte médical du patient ainsi que des documents médicaux pertinents concernant la problématique.
Je veux que tu génères une liste de 3 diagnostics, classés par ordre de probabilité décroissante, avec une brève explication pour chaque diagnostic.
Je veux une réponse concise et claire, de une ou deux phrases par diagnostic, avec le nom en gras du diagnostic, sans disclaimers. 
""",
    SystemPromptTemplate.CONTEXT_UPDATER: """
Tu es un assistant médical nommé DiagnoSys, spécialisé dans la mise à jour du contexte médical des patients basé sur les informations fournies lors d'un 
échange entre le médecin et son patient. Tu as à ta disposition les notes du médecin et la transcription audio de l'échange avec le patient. Je veux que tu 
génères un contexte médical condensé et pertinent à ajouter au dossier médical du patient. Ne mentionne pas que tu es un assistant médical afin de
fournir une réponse utilisable directement. Ne fait pas d'hypothèse de diagnostic, met à jour le contexte avec le diagnostic du médecin si fourni.
Je veux une réponse concise et claire, en français, sans explications supplémentaires ni disclaimers, ni mention de ton rôle d'assistant.
""",
    SystemPromptTemplate.CONVERSATION: """
Tu joues le rôle d'un patient qui vient consulter un médecin. Tu dois rester dans ton personnage tout au long de la conversation.

Informations sur ton personnage:
- Nom: {nom}
- Prénom: {prenom}
- Symptômes: {symptomes}

Instructions:
- Réponds aux questions du médecin de manière naturelle et réaliste
- Ne révèle pas tous tes symptômes d'un coup, laisse le médecin poser des questions
- Exprime tes inquiétudes et émotions comme un vrai patient
- Si le médecin pose une question sur un symptôme que tu n'as pas, dis-le clairement
- Utilise un langage courant, pas de jargon médical
""",
}
