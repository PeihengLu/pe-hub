from typing import Optional
import torch

from pe_common.model_interface import BasePEModel
from .deepprime_wrapper import DeepPrimeModelWrapper
from .oped_wrapper import OPEDModelWrapper
from .pridict2_wrapper import PRIDICT2ModelWrapper


class ModelFactory:
    """Factory for creating model wrappers"""
    
    _models = {
        'deepprime': DeepPrimeModelWrapper,
        'oped': OPEDModelWrapper,
        'pridict2': PRIDICT2ModelWrapper,
    }
    
    @classmethod
    def create_model(cls, model_name: str, device: Optional[torch.device] = None, **kwargs) -> BasePEModel:
        """
        Create a model wrapper instance
        
        Args:
            model_name: Name of the model ('deepprime', 'oped', 'pridict', 'pridict2', etc.)
            device: PyTorch device
            **kwargs: Additional model-specific parameters
                For DeepPrime: pe_system='PE2max', cell_type='HEK293T'
                For PRIDICT/PRIDICT2: wsize=20
            
        Returns:
            BasePEModel instance
            
        Examples:
            >>> # Create DeepPrime model
            >>> model = ModelFactory.create_model('deepprime', pe_system='PE2max', cell_type='HEK293T')
            
            >>> # Create OPED model
            >>> model = ModelFactory.create_model('oped')
        """
        model_name_lower = model_name.lower()
        
        if model_name_lower not in cls._models:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(cls._models.keys())}"
            )
        
        model_class = cls._models[model_name_lower]
        return model_class(device=device, **kwargs)
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """
        Register a new model wrapper
        
        Args:
            name: Model name (will be converted to lowercase)
            model_class: Model wrapper class (must inherit from BasePEModel)
        """
        from pe_common.model_interface import BasePEModel
        
        if not issubclass(model_class, BasePEModel):
            raise TypeError(f"{model_class} must inherit from BasePEModel")
        
        cls._models[name.lower()] = model_class
    
    @classmethod
    def list_models(cls) -> list:
        """List all available models"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        model_name_lower = model_name.lower()
        
        if model_name_lower not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = cls._models[model_name_lower]
        
        info = {
            'name': model_name_lower,
            'class': model_class.__name__,
            'module': model_class.__module__,
        }
        
        # Add model-specific info if available
        if model_name_lower == 'deepprime':
            info['supported_pe_systems'] = DeepPrimeModelWrapper.SUPPORTED_PE_SYSTEMS
            info['supported_cell_types'] = DeepPrimeModelWrapper.SUPPORTED_CELL_TYPES
        elif model_name_lower == 'pridict2':
            info['supported_cell_types'] = PRIDICT2ModelWrapper.SUPPORTED_CELL_TYPES
            info['supported_model_names'] = PRIDICT2ModelWrapper.SUPPORTED_MODEL_NAMES
            info['supported_models_map'] = PRIDICT2ModelWrapper.get_supported_models()
        
        return info
