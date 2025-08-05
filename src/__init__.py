"""
LLM Evaluation Framework

A comprehensive framework for evaluating and comparing Large Language Models.
"""

__version__ = "1.0.0"
__author__ = "Mark Stent"

from .models import ModelResponse, TestCase, EvaluationResult
from .llm_models import BaseModel, OpenAIModel, AnthropicModel, GeminiModel, OllamaModel
from .config import ConfigManager, ConfigValidator
from .metrics import AdvancedMetrics, CategorySpecificMetrics
from .utils import setup_logging, PerformanceMonitor
from .exceptions import (
    LLMEvaluatorError,
    ModelConfigError,
    APIError,
    EvaluationError,
    TestSuiteError,
    MetricsError
)

__all__ = [
    "ModelResponse",
    "TestCase", 
    "EvaluationResult",
    "BaseModel",
    "OpenAIModel",
    "AnthropicModel", 
    "GeminiModel",
    "OllamaModel",
    "ConfigManager",
    "ConfigValidator",
    "AdvancedMetrics",
    "CategorySpecificMetrics",
    "setup_logging",
    "PerformanceMonitor",
    "LLMEvaluatorError",
    "ModelConfigError",
    "APIError",
    "EvaluationError",
    "TestSuiteError",
    "MetricsError"
]
