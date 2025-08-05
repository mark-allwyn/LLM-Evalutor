"""
Data models for the LLM evaluation framework.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class ModelResponse:
    """Response from an LLM model."""
    text: str
    tokens_used: Optional[int] = None
    latency: float = 0.0
    cost: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class TestCase:
    """Individual test case for evaluation."""
    id: str
    prompt: str
    expected_output: Optional[str] = None
    category: str = "general"
    metadata: Dict[str, Any] = None


@dataclass
class EvaluationResult:
    """Result of evaluating a model on a test case."""
    model_name: str
    test_case_id: str
    category: str
    prompt: str
    response: str
    expected_output: Optional[str]
    latency: float
    tokens_used: Optional[int]
    cost: Optional[float]
    metrics: Dict[str, float]
    timestamp: str
    error: Optional[str] = None
