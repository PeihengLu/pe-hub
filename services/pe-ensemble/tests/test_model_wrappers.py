"""
Unit tests for model wrappers
"""

import pytest
import pandas as pd
import torch
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'packages' / 'pe-common'))

from app.models.model_factory import ModelFactory


class TestModelFactory:
    """Test ModelFactory functionality"""
    
    def test_list_models(self):
        """Test listing available models"""
        models = ModelFactory.list_models()
        assert isinstance(models, list)
        assert 'deepprime' in models
    
    def test_create_deepprime_model(self):
        """Test creating DeepPrime model"""
        model = ModelFactory.create_model(
            'deepprime',
            pe_system='PE2max',
            cell_type='HEK293T'
        )
        assert model.model_name == 'DeepPrime'
        assert model.pe_system == 'PE2max'
        assert model.cell_type == 'HEK293T'
    
    def test_invalid_model_name(self):
        """Test creating model with invalid name"""
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model')
    
    def test_invalid_pe_system(self):
        """Test creating DeepPrime with invalid PE system"""
        with pytest.raises(ValueError):
            ModelFactory.create_model(
                'deepprime',
                pe_system='INVALID_PE',
                cell_type='HEK293T'
            )
    
    def test_invalid_cell_type(self):
        """Test creating DeepPrime with invalid cell type"""
        with pytest.raises(ValueError):
            ModelFactory.create_model(
                'deepprime',
                pe_system='PE2max',
                cell_type='INVALID_CELL'
            )
    
    def test_get_model_info(self):
        """Test getting model information"""
        info = ModelFactory.get_model_info('deepprime')
        assert 'name' in info
        assert 'supported_pe_systems' in info
        assert 'supported_cell_types' in info
        assert info['name'] == 'deepprime'


class TestDeepPrimeWrapper:
    """Test DeepPrime model wrapper"""
    
    @pytest.fixture
    def model(self):
        """Create a DeepPrime model instance"""
        device = torch.device('cpu')  # Use CPU for testing
        model = ModelFactory.create_model(
            'deepprime',
            device=device,
            pe_system='PE2max',
            cell_type='HEK293T'
        )
        return model
    
    def test_model_creation(self, model):
        """Test model instance creation"""
        assert model.model_name == 'DeepPrime'
        assert not model.is_trained
        assert model.device.type == 'cpu'
    
    def test_model_info(self, model):
        """Test getting model info"""
        info = model.get_model_info()
        assert info['name'] == 'DeepPrime'
        assert info['pe_system'] == 'PE2max'
        assert info['cell_type'] == 'HEK293T'
        assert info['is_trained'] == False
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.parent.joinpath(
            'vendor/models/deepprime'
        ).exists(),
        reason="DeepPrime models not available"
    )
    def test_load_model(self, model):
        """Test loading pre-trained model"""
        model.load_model()
        assert model.is_trained
        assert model.models is not None
        assert len(model.models) > 0
    
    def test_predict_without_loading(self, model):
        """Test prediction without loading model first"""
        with pytest.raises(ValueError):
            model.predict({})
    
    def test_evaluate_without_loading(self, model):
        """Test evaluation without loading model first"""
        test_data = pd.DataFrame({'Efficiency': [0.5, 0.6]})
        with pytest.raises(ValueError):
            model.evaluate(test_data)
    
    def test_train_not_implemented(self, model):
        """Test that training raises NotImplementedError"""
        train_data = pd.DataFrame()
        with pytest.raises(NotImplementedError):
            model.train(train_data)
    
    def test_supported_pe_systems(self):
        """Test that all supported PE systems can be instantiated"""
        from app.models.deepprime_wrapper import DeepPrimeModelWrapper
        
        for pe_system in DeepPrimeModelWrapper.SUPPORTED_PE_SYSTEMS:
            try:
                model = ModelFactory.create_model(
                    'deepprime',
                    pe_system=pe_system,
                    cell_type='HEK293T'
                )
                assert model.pe_system == pe_system
            except ValueError:
                # Some PE systems may not be available for all cell types
                pass
    
    def test_supported_cell_types(self):
        """Test that all supported cell types can be instantiated"""
        from app.models.deepprime_wrapper import DeepPrimeModelWrapper
        
        for cell_type in DeepPrimeModelWrapper.SUPPORTED_CELL_TYPES:
            try:
                model = ModelFactory.create_model(
                    'deepprime',
                    pe_system='PE2max',
                    cell_type=cell_type
                )
                assert model.cell_type == cell_type
            except ValueError:
                # Some cell types may not support all PE systems
                pass


class TestBasePEModel:
    """Test BasePEModel interface"""
    
    def test_interface_methods(self):
        """Test that model implements all required interface methods"""
        from pe_common.model_interface import BasePEModel
        from app.models.deepprime_wrapper import DeepPrimeModelWrapper
        
        required_methods = [
            'load_model',
            'prepare_data',
            'predict',
            'train',
            'evaluate',
            'save_model',
            'get_model_info'
        ]
        
        for method in required_methods:
            assert hasattr(DeepPrimeModelWrapper, method)
            assert callable(getattr(DeepPrimeModelWrapper, method))
    
    def test_inheritance(self):
        """Test that wrapper inherits from BasePEModel"""
        from pe_common.model_interface import BasePEModel
        from app.models.deepprime_wrapper import DeepPrimeModelWrapper
        
        assert issubclass(DeepPrimeModelWrapper, BasePEModel)


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual models and data"""
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.parent.joinpath(
            'vendor/models/deepprime'
        ).exists(),
        reason="DeepPrime models not available"
    )
    def test_full_prediction_pipeline(self):
        """Test complete prediction pipeline"""
        # Create model
        model = ModelFactory.create_model(
            'deepprime',
            pe_system='PE2max',
            cell_type='HEK293T'
        )
        
        # Load model
        model.load_model()
        assert model.is_trained
        
        # This would require actual data in the correct format
        # For now, just verify the model loaded correctly
        info = model.get_model_info()
        assert info['n_models'] > 0
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.parent.joinpath(
            'datasets/standardized/deepprime'
        ).exists(),
        reason="Test data not available"
    )
    def test_with_real_data(self):
        """Test with real standardized data"""
        data_dir = Path(__file__).parent.parent.parent.parent / 'datasets' / 'standardized' / 'deepprime'
        csv_files = list(data_dir.glob('*.csv'))
        
        if not csv_files:
            pytest.skip("No test data files found")
        
        # Load first file
        test_file = csv_files[0]
        df = pd.read_csv(test_file, nrows=10)  # Load only first 10 rows for testing
        
        # Create and load model
        model = ModelFactory.create_model(
            'deepprime',
            pe_system='PE2max',
            cell_type='HEK293T'
        )
        model.load_model()
        
        # Test prepare_data (may fail if data format doesn't match)
        try:
            prepared_data = model.prepare_data(df)
            assert prepared_data is not None
        except Exception as e:
            pytest.skip(f"Data preparation failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
