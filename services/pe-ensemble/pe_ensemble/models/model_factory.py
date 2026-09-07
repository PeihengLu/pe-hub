from typing import Optional
import torch

from pe_common.model_interface import BasePEModel
from .registry import model_registry, ModelSpec


class ModelFactory:
    """Factory for creating model wrappers"""

    @classmethod
    def create_model(
        cls,
        model_name: str,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> BasePEModel:
        """
        Create a model wrapper instance

        Args:
            model_name: Name of the model ('deepprime', 'oped', 'pridict2', etc.)
            device: PyTorch device
            **kwargs: Additional model-specific parameters
                For DeepPrime: pe_system='PE2max', cell_type='HEK293T'
                For PRIDICT/PRIDICT2: wsize=20

        Returns:
            BasePEModel instance
        """
        spec = model_registry.get(model_name)
        return spec.wrapper_class(device=device, **kwargs)

    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """
        Register a new model wrapper at runtime (plugin loader).

        For full metadata (PE-DB format, weight format, etc.) use
        ``model_registry.register`` with a complete ``ModelSpec``.
        """
        if not issubclass(model_class, BasePEModel):
            raise TypeError(f"{model_class} must inherit from BasePEModel")

        key = name.strip().lower()
        if model_registry.is_registered(key):
            raise ValueError(f"Model '{key}' is already registered.")

        model_registry.register(
            ModelSpec(
                name=key,
                wrapper_class=model_class,
                display_name=key,
                description=f"Plugin model: {key}",
                model_type="unknown",
                pe_db_format=key,
                weight_format=f"{key}_weights",
                source="plugin",
            )
        )

    @classmethod
    def list_models(cls) -> list:
        """List all available models"""
        return list(model_registry.names())

    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """
        Get information about a specific model

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        spec = model_registry.get(model_name)
        from . import weights_registry

        return {
            "name": spec.name,
            "class": spec.wrapper_class.__name__,
            "module": spec.wrapper_class.__module__,
            "available_weights": weights_registry.list_weight_ids(spec.name),
            "pe_db_format": spec.pe_db_format,
            "weight_format": spec.weight_format,
            "source": spec.source,
        }
