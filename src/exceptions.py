"""
Custom exceptions for the LLM evaluation framework.
"""


class LLMEvaluatorError(Exception):
    """Base exception for LLM evaluator errors."""
    pass


class ModelConfigError(LLMEvaluatorError):
    """Raised when there's an error in model configuration."""
    pass


class APIError(LLMEvaluatorError):
    """Raised when there's an API error from LLM providers."""
    pass


class EvaluationError(LLMEvaluatorError):
    """Raised when there's an error during evaluation."""
    pass


class TestSuiteError(LLMEvaluatorError):
    """Raised when there's an error with the test suite."""
    pass


class MetricsError(LLMEvaluatorError):
    """Raised when there's an error calculating metrics."""
    pass
