"""
Unit tests for the LLM evaluation framework.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json
import tempfile
from pathlib import Path

from src.models import TestCase, ModelResponse, EvaluationResult
from src.llm_models import OpenAIModel, AnthropicModel
from src.config import ConfigManager, ConfigValidator
from src.exceptions import ModelConfigError


class TestConfigManager:
    """Test configuration management."""
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = ConfigManager.create_default_config()
        
        assert 'models' in config
        assert 'pricing' in config
        assert 'wandb_project' in config
        assert isinstance(config['models'], dict)
        assert isinstance(config['pricing'], dict)
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_data = ConfigManager.create_default_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = ConfigManager.load_config(config_path)
            assert loaded_config == config_data
        finally:
            Path(config_path).unlink()
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(ModelConfigError):
            ConfigManager.load_config('nonexistent.yaml')


class TestConfigValidator:
    """Test configuration validation."""
    
    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        config = ConfigManager.create_default_config()
        # Should not raise any exception
        ConfigValidator.validate_config(config)
    
    def test_validate_missing_models_section(self):
        """Test validation with missing models section."""
        config = {'pricing': {}}
        with pytest.raises(ModelConfigError, match="Missing required configuration section: models"):
            ConfigValidator.validate_config(config)
    
    def test_validate_missing_pricing_section(self):
        """Test validation with missing pricing section."""
        config = {'models': {}}
        with pytest.raises(ModelConfigError, match="Missing required configuration section: pricing"):
            ConfigValidator.validate_config(config)
    
    def test_validate_empty_models(self):
        """Test validation with empty models configuration."""
        config = {
            'models': {},
            'pricing': {'default_pricing': {'input': 1.0, 'output': 2.0}}
        }
        with pytest.raises(ModelConfigError, match="No models configured"):
            ConfigValidator.validate_config(config)
    
    def test_validate_unsupported_provider(self):
        """Test validation with unsupported provider."""
        config = {
            'models': {
                'unsupported_provider': [{'name': 'test-model'}]
            },
            'pricing': {'default_pricing': {'input': 1.0, 'output': 2.0}}
        }
        with pytest.raises(ModelConfigError, match="Unsupported provider: unsupported_provider"):
            ConfigValidator.validate_config(config)


class TestModels:
    """Test model classes."""
    
    def test_model_response_creation(self):
        """Test creating a ModelResponse."""
        response = ModelResponse(
            text="Test response",
            tokens_used=10,
            latency=0.5,
            cost=0.01
        )
        
        assert response.text == "Test response"
        assert response.tokens_used == 10
        assert response.latency == 0.5
        assert response.cost == 0.01
    
    def test_test_case_creation(self):
        """Test creating a TestCase."""
        test_case = TestCase(
            id="test_1",
            prompt="What is 2+2?",
            expected_output="4",
            category="math"
        )
        
        assert test_case.id == "test_1"
        assert test_case.prompt == "What is 2+2?"
        assert test_case.expected_output == "4"
        assert test_case.category == "math"
    
    def test_evaluation_result_creation(self):
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            model_name="gpt-4o-mini",
            test_case_id="test_1",
            category="math",
            prompt="What is 2+2?",
            response="4",
            expected_output="4",
            latency=0.5,
            tokens_used=10,
            cost=0.01,
            metrics={'bleu': 1.0},
            timestamp="2024-01-01T00:00:00"
        )
        
        assert result.model_name == "gpt-4o-mini"
        assert result.test_case_id == "test_1"
        assert result.metrics['bleu'] == 1.0


class TestLLMModels:
    """Test LLM model implementations."""
    
    def test_openai_model_initialization(self):
        """Test OpenAI model initialization."""
        pricing_config = {
            'openai': {
                'gpt-4o-mini': {'input': 0.15, 'output': 0.60}
            },
            'default_pricing': {'input': 1.0, 'output': 3.0}
        }
        
        model = OpenAIModel("gpt-4o-mini", "test-key", pricing_config)
        
        assert model.model_name == "gpt-4o-mini"
        assert model._cost_per_token == {'input': 0.15, 'output': 0.60}
    
    def test_cost_calculation(self):
        """Test cost calculation."""
        pricing_config = {
            'openai': {
                'gpt-4o-mini': {'input': 0.15, 'output': 0.60}
            }
        }
        
        model = OpenAIModel("gpt-4o-mini", "test-key", pricing_config)
        cost = model.calculate_cost(1000, 500)  # 1000 input tokens, 500 output tokens
        
        expected_cost = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
        assert abs(cost - expected_cost) < 1e-10
    
    @pytest.mark.asyncio
    async def test_openai_model_generate_mock(self):
        """Test OpenAI model generation with mocked API."""
        pricing_config = {
            'openai': {
                'gpt-4o-mini': {'input': 0.15, 'output': 0.60}
            }
        }
        
        model = OpenAIModel("gpt-4o-mini", "test-key", pricing_config)
        
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4o-mini"
        
        with patch.object(model.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            response = await model.generate("Test prompt")
            
            assert response.text == "Test response"
            assert response.tokens_used == 15
            assert response.cost > 0
            assert response.latency > 0


if __name__ == "__main__":
    pytest.main([__file__])
