"""
Configuration management and validation for the LLM evaluation framework.
"""
import os
import yaml
from typing import Dict, Any, List
from pathlib import Path

from .exceptions import ModelConfigError


class ConfigValidator:
    """Validates configuration files and settings."""
    
    REQUIRED_SECTIONS = ['models', 'pricing']
    REQUIRED_MODEL_FIELDS = ['name']
    SUPPORTED_PROVIDERS = ['openai', 'anthropic', 'gemini', 'ollama']
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate the entire configuration."""
        ConfigValidator._validate_structure(config)
        ConfigValidator._validate_models(config.get('models', {}))
        ConfigValidator._validate_pricing(config.get('pricing', {}))
        ConfigValidator._validate_environment_variables(config.get('models', {}))
    
    @staticmethod
    def _validate_structure(config: Dict[str, Any]) -> None:
        """Validate the basic structure of the configuration."""
        for section in ConfigValidator.REQUIRED_SECTIONS:
            if section not in config:
                raise ModelConfigError(f"Missing required configuration section: {section}")
    
    @staticmethod
    def _validate_models(models_config: Dict[str, Any]) -> None:
        """Validate the models configuration."""
        if not models_config:
            raise ModelConfigError("No models configured")
        
        for provider, models in models_config.items():
            if provider not in ConfigValidator.SUPPORTED_PROVIDERS:
                raise ModelConfigError(f"Unsupported provider: {provider}")
            
            if not isinstance(models, list):
                raise ModelConfigError(f"Models for {provider} must be a list")
            
            for model in models:
                ConfigValidator._validate_model(model, provider)
    
    @staticmethod
    def _validate_model(model: Dict[str, Any], provider: str) -> None:
        """Validate individual model configuration."""
        for field in ConfigValidator.REQUIRED_MODEL_FIELDS:
            if field not in model:
                raise ModelConfigError(f"Missing required field '{field}' for {provider} model")
        
        # Provider-specific validation
        if provider in ['openai', 'anthropic', 'gemini']:
            if 'api_key' not in model:
                raise ModelConfigError(f"Missing API key for {provider} model {model['name']}")
        elif provider == 'ollama':
            if 'base_url' not in model:
                model['base_url'] = 'http://localhost:11434'  # Set default
    
    @staticmethod
    def _validate_pricing(pricing_config: Dict[str, Any]) -> None:
        """Validate the pricing configuration."""
        if not pricing_config:
            raise ModelConfigError("No pricing configuration found")
        
        # Check if default_pricing exists
        if 'default_pricing' not in pricing_config:
            raise ModelConfigError("Missing default_pricing in configuration")
    
    @staticmethod
    def _validate_environment_variables(models_config: Dict[str, Any]) -> None:
        """Validate that required environment variables are set."""
        required_env_vars = set()
        
        for provider, models in models_config.items():
            for model in models:
                api_key = model.get('api_key', '')
                if api_key.startswith('${') and api_key.endswith('}'):
                    env_var = api_key[2:-1]
                    required_env_vars.add(env_var)
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ModelConfigError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                f"Please set these in your .env file or environment."
            )


class ConfigManager:
    """Manages configuration loading and creation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise ModelConfigError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ModelConfigError(f"Invalid YAML configuration: {e}")
        
        # Validate configuration
        ConfigValidator.validate_config(config)
        
        return config
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """Create default configuration with current pricing."""
        return {
            'wandb_project': 'llm-evaluation',
            'wandb_entity': None,
            'max_concurrent': 5,
            'max_tokens': 1000,
            'temperature': 0.7,
            'models': {
                'openai': [
                    {'name': 'gpt-4o-mini', 'api_key': '${OPENAI_API_KEY}'},
                    {'name': 'gpt-3.5-turbo', 'api_key': '${OPENAI_API_KEY}'}
                ],
                'anthropic': [
                    {'name': 'claude-3-haiku-20240307', 'api_key': '${ANTHROPIC_API_KEY}'}
                ],
                'gemini': [
                    {'name': 'gemini-1.5-flash', 'api_key': '${GOOGLE_API_KEY}'}
                ],
                'ollama': [
                    {'name': 'llama3.2', 'base_url': 'http://localhost:11434'}
                ]
            },
            'pricing': {
                'openai': {
                    'gpt-4o': {'input': 3.00, 'output': 10.00},
                    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
                    'gpt-3.5-turbo': {'input': 1.50, 'output': 2.00}
                },
                'anthropic': {
                    'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
                    'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00}
                },
                'gemini': {
                    'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
                    'gemini-1.5-flash': {'input': 0.075, 'output': 0.30}
                },
                'ollama': {
                    'default': {'input': 0.0, 'output': 0.0}
                },
                'default_pricing': {'input': 1.00, 'output': 3.00}
            }
        }
