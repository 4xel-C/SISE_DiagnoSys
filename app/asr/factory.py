"""Factory for ASR services."""

import logging
import os

from dotenv import load_dotenv

from app.asr.base_asr import ASRServiceBase

logger = logging.getLogger(__name__)

load_dotenv()

type ASRService = type[ASRServiceBase]
type ASRAnswer = dict[str, str | bool]


class ASRServiceFactory:
    """
    Factory/registry for ASR services.

    Register implementations with:
        @ASRServiceFactory.register("sherpa_onnx")
        class SherpaOnnxASRService(ASRServiceBase): ...
    """

    _registry: dict[str, ASRService] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an ASR service class under a given name."""
        key = name.lower().strip()

        def decorator(service_cls: ASRService) -> ASRService:
            if key in cls._registry:
                raise ValueError(f"ASR service '{key}' is already registered.")
            cls._registry[key] = service_cls
            return service_cls

        return decorator

    @classmethod
    def create(cls, which: str = "sherpa_onnx") -> ASRServiceBase:
        """
        Create an ASR service instance based on the specified type or environment configuration.

        Args:
            which: "kyutai" / "sherpa_onnx" / etc.

        Raises:
            ValueError: If service name is unknown.
            NotImplementedError: If the requested service is not registered.

        Returns:
            ASRServiceBase: An instance of the requested ASR service.
        """
        key = which.lower().strip()
        service_cls = cls._registry.get(key)
        if service_cls is None:
            logger.error(
                "ASR service '%s' is not registered. Available services: %s",
                key,
                sorted(cls._registry.keys()),
            )
            raise NotImplementedError(
                f"ASR service '{key}' is not registered. Registered services: {sorted(cls._registry.keys())}"
            )
        return service_cls()

    @classmethod
    def available(cls, which: str = "sherpa_onnx") -> bool:
        """Check if the specified ASR service is available."""
        try:
            service = cls.create(which)
            return service.is_available()
        except (
            NotImplementedError,
            ValueError,
            OSError,
            RuntimeError,
            FileNotFoundError,
        ):
            logger.error("ASR service '%s' is not available.", which)
            return False
