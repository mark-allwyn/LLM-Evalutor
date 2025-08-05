# main.py - Complete LLM Evaluation Application
import asyncio
import json
import os
import time
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import argparse
import logging
from datetime import datetime

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Core dependencies
import pandas as pd
import numpy as np
import wandb
from tqdm.asyncio import tqdm
import aiohttp
import openai
from openai import AsyncOpenAI
import anthropic
import google.generativeai as genai

# Evaluation metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score
from sentence_transformers import SentenceTransformer
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ModelResponse:
    text: str
    tokens_used: Optional[int] = None
    latency: float = 0.0
    cost: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class TestCase:
    id: str
    prompt: str
    expected_output: Optional[str] = None
    category: str = "general"
    metadata: Dict[str, Any] = None

@dataclass
class EvaluationResult:
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

# =============================================================================
# Base Model Interface
# =============================================================================

class BaseModel(ABC):
    def __init__(self, model_name: str, pricing_config: Dict[str, Any] = None, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self.pricing_config = pricing_config or {}
        self._cost_per_token = self.get_cost_per_token()
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> Dict[str, float]:
        pass
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = input_tokens / 1_000_000 * self._cost_per_token.get('input', 0)  # Per million tokens
        output_cost = output_tokens / 1_000_000 * self._cost_per_token.get('output', 0)  # Per million tokens
        return input_cost + output_cost

# =============================================================================
# Model Implementations
# =============================================================================

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str, pricing_config: Dict[str, Any] = None, **kwargs):
        super().__init__(model_name, pricing_config, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            latency = time.time() - start_time
            
            # Calculate cost
            usage = response.usage
            cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens) if usage else None
            
            return ModelResponse(
                text=response.choices[0].message.content,
                tokens_used=usage.total_tokens if usage else None,
                latency=latency,
                cost=cost,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'model': response.model
                }
            )
        except Exception as e:
            logger.error(f"OpenAI API error for {self.model_name}: {e}")
            raise
    
    def get_cost_per_token(self) -> Dict[str, float]:
        # Get pricing from config, fallback to default
        openai_pricing = self.pricing_config.get('openai', {})
        model_pricing = openai_pricing.get(self.model_name, {})
        
        if model_pricing:
            return model_pricing
        
        # Fallback to default pricing if model not found in config
        default_pricing = self.pricing_config.get('default_pricing', {'input': 1.00, 'output': 3.00})
        logger.warning(f"No pricing found for {self.model_name}, using default: {default_pricing}")
        return default_pricing

class AnthropicModel(BaseModel):
    def __init__(self, model_name: str, api_key: str, pricing_config: Dict[str, Any] = None, **kwargs):
        super().__init__(model_name, pricing_config, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            
            latency = time.time() - start_time
            
            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self.calculate_cost(input_tokens, output_tokens)
            
            return ModelResponse(
                text=response.content[0].text,
                tokens_used=input_tokens + output_tokens,
                latency=latency,
                cost=cost,
                metadata={
                    'stop_reason': response.stop_reason,
                    'model': response.model
                }
            )
        except Exception as e:
            logger.error(f"Anthropic API error for {self.model_name}: {e}")
            raise
    
    def get_cost_per_token(self) -> Dict[str, float]:
        # Get pricing from config, fallback to default
        anthropic_pricing = self.pricing_config.get('anthropic', {})
        model_pricing = anthropic_pricing.get(self.model_name, {})
        
        if model_pricing:
            return model_pricing
        
        # Fallback to default pricing if model not found in config
        default_pricing = self.pricing_config.get('default_pricing', {'input': 3.00, 'output': 15.00})
        logger.warning(f"No pricing found for {self.model_name}, using default: {default_pricing}")
        return default_pricing

class GeminiModel(BaseModel):
    def __init__(self, model_name: str, api_key: str, pricing_config: Dict[str, Any] = None, **kwargs):
        super().__init__(model_name, pricing_config, **kwargs)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
            )
            
            # Generate response (Gemini API is not async, so we run in thread)
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(
                    prompt, 
                    generation_config=generation_config
                )
            )
            
            latency = time.time() - start_time
            
            # Extract text from response
            response_text = response.text if response.text else ""
            
            # Calculate tokens and cost
            # Note: Gemini API doesn't always provide token counts in response
            tokens_used = None
            cost = None
            
            # Try to get usage metadata if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                tokens_used = prompt_tokens + output_tokens
                cost = self.calculate_cost(prompt_tokens, output_tokens)
            
            return ModelResponse(
                text=response_text,
                tokens_used=tokens_used,
                latency=latency,
                cost=cost,
                metadata={
                    'finish_reason': getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None,
                    'model': self.model_name,
                    'safety_ratings': [rating for candidate in response.candidates for rating in candidate.safety_ratings] if response.candidates else []
                }
            )
        except Exception as e:
            logger.error(f"Gemini API error for {self.model_name}: {e}")
            raise
    
    def get_cost_per_token(self) -> Dict[str, float]:
        # Get pricing from config, fallback to default
        gemini_pricing = self.pricing_config.get('gemini', {})
        model_pricing = gemini_pricing.get(self.model_name, {})
        
        if model_pricing:
            return model_pricing
        
        # Fallback to default pricing if model not found in config
        default_pricing = self.pricing_config.get('default_pricing', {'input': 0.50, 'output': 1.50})
        logger.warning(f"No pricing found for {self.model_name}, using default: {default_pricing}")
        return default_pricing

class OllamaModel(BaseModel):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", pricing_config: Dict[str, Any] = None, **kwargs):
        super().__init__(model_name, pricing_config, **kwargs)
        self.base_url = base_url.rstrip('/')
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', 0.7),
                        "num_predict": kwargs.get('max_tokens', 1000)
                    }
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama API error: {response.status}")
                    
                    result = await response.json()
                    latency = time.time() - start_time
                    
                    return ModelResponse(
                        text=result.get('response', ''),
                        tokens_used=result.get('eval_count', 0) + result.get('prompt_eval_count', 0),
                        latency=latency,
                        cost=0.0,  # Ollama is free
                        metadata={
                            'eval_count': result.get('eval_count'),
                            'prompt_eval_count': result.get('prompt_eval_count')
                        }
                    )
        except Exception as e:
            logger.error(f"Ollama API error for {self.model_name}: {e}")
            raise
    
    def get_cost_per_token(self) -> Dict[str, float]:
        # Ollama models are typically free, but check config for custom pricing
        ollama_pricing = self.pricing_config.get('ollama', {})
        model_pricing = ollama_pricing.get(self.model_name)
        
        if model_pricing:
            return model_pricing
        
        # Default to free for Ollama models
        return ollama_pricing.get('default', {'input': 0.0, 'output': 0.0})

# =============================================================================
# Evaluation Metrics
# =============================================================================

class EvaluationMetrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        if not reference or not candidate:
            return 0.0
        
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()
        
        smoothing_function = SmoothingFunction().method4
        return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not reference or not candidate:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_bert_score(self, reference: str, candidate: str) -> float:
        """Calculate BERTScore"""
        if not reference or not candidate:
            return 0.0
        
        try:
            P, R, F1 = bert_score.score([candidate], [reference], lang='en')
            return F1.item()
        except:
            return 0.0
    
    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        if not reference or not candidate:
            return 0.0
        
        embeddings = self.sentence_model.encode([reference, candidate])
        return np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
    
    def calculate_length_ratio(self, reference: str, candidate: str) -> float:
        """Calculate length ratio between candidate and reference"""
        if not reference:
            return 1.0 if not candidate else float('inf')
        return len(candidate.split()) / len(reference.split())
    
    def calculate_all_metrics(self, reference: Optional[str], candidate: str) -> Dict[str, float]:
        """Calculate all available metrics"""
        # Initialize all metrics with default values
        metrics = {
            'bleu': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'bert_score': 0.0,
            'semantic_similarity': 0.0,
            'length_ratio': 0.0,
            'length': len(candidate.split()),
            'char_count': len(candidate)
        }
        
        if reference is None:
            # For creative tasks without expected output, keep default 0.0 values
            return metrics
        
        # Calculate actual metrics when reference is available
        metrics['bleu'] = self.calculate_bleu(reference, candidate)
        rouge_scores = self.calculate_rouge(reference, candidate)
        metrics.update(rouge_scores)
        metrics['bert_score'] = self.calculate_bert_score(reference, candidate)
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(reference, candidate)
        metrics['length_ratio'] = self.calculate_length_ratio(reference, candidate)
        
        return metrics

# =============================================================================
# Test Suite Management
# =============================================================================

class TestSuite:
    def __init__(self, test_cases: List[TestCase]):
        self.test_cases = test_cases
    
    @classmethod
    def from_file(cls, filepath: str) -> 'TestSuite':
        """Load test cases from file (JSON, JSONL, or CSV)"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                test_cases = [TestCase(**item) for item in data]
        
        elif filepath.suffix == '.jsonl':
            test_cases = []
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    test_cases.append(TestCase(**data))
        
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
            test_cases = []
            for _, row in df.iterrows():
                test_cases.append(TestCase(
                    id=row['id'],
                    prompt=row['prompt'],
                    expected_output=row.get('expected_output'),
                    category=row.get('category', 'general'),
                    metadata=json.loads(row.get('metadata', '{}'))
                ))
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls(test_cases)
    
    @classmethod
    def create_sample_suite(cls) -> 'TestSuite':
        """Create a sample test suite for demonstration"""
        test_cases = [
            TestCase(
                id="math_1",
                prompt="What is 127 * 83?",
                expected_output="10541",
                category="math"
            ),
            TestCase(
                id="reasoning_1",
                prompt="If a train leaves New York at 3 PM traveling at 60 mph, and another train leaves Boston at 4 PM traveling at 80 mph, and the distance between the cities is 200 miles, when will they meet?",
                expected_output="They will meet at approximately 5:26 PM.",
                category="reasoning"
            ),
            TestCase(
                id="creative_1",
                prompt="Write a haiku about artificial intelligence.",
                category="creative"
            ),
            TestCase(
                id="factual_1",
                prompt="What is the capital of Australia?",
                expected_output="Canberra",
                category="factual"
            ),
            TestCase(
                id="coding_1",
                prompt="Write a Python function to check if a string is a palindrome.",
                category="coding"
            )
        ]
        
        return cls(test_cases)
    
    def filter_by_category(self, category: str) -> 'TestSuite':
        """Filter test cases by category"""
        filtered_cases = [tc for tc in self.test_cases if tc.category == category]
        return TestSuite(filtered_cases)
    
    def sample(self, n: int) -> 'TestSuite':
        """Sample n test cases randomly"""
        import random
        sampled_cases = random.sample(self.test_cases, min(n, len(self.test_cases)))
        return TestSuite(sampled_cases)

# =============================================================================
# Main Evaluator
# =============================================================================

class ModelEvaluator:
    def __init__(self, models: List[BaseModel], test_suite: TestSuite, config: Dict[str, Any]):
        self.models = models
        self.test_suite = test_suite
        self.config = config
        self.metrics_calculator = EvaluationMetrics()
        self.results: List[EvaluationResult] = []
    
    async def run_evaluation(self, experiment_name: str = None) -> pd.DataFrame:
        """Run evaluation across all models and test cases"""
        if experiment_name is None:
            experiment_name = f"llm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        

        wandb_entity  = os.getenv("WANDB_ENTITY") or self.config.get('wandb_entity')
        wandb_project = os.getenv("WANDB_PROJECT") or self.config.get('wandb_project', 'llm-evaluation')
        print(f"[DEBUG] W&B entity={wandb_entity!r}, project={wandb_project!r}")
        # Initialize W&B run
        
        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),                 # ← add this
            project=os.getenv("WANDB_PROJECT", "llm-evaluation"),
            config={
                'models': [model.model_name for model in self.models],
                'test_cases_count': len(self.test_suite.test_cases),
                'categories': list(set(tc.category for tc in self.test_suite.test_cases)),
                **self.config
            }
        )
        
        logger.info(f"Starting evaluation with {len(self.models)} models and {len(self.test_suite.test_cases)} test cases")
        
        # Create evaluation tasks
        tasks = []
        for model in self.models:
            for test_case in self.test_suite.test_cases:
                tasks.append(self._evaluate_single(model, test_case))
        
        # Run evaluations with progress bar
        semaphore = asyncio.Semaphore(self.config.get('max_concurrent', 5))
        
        async def bounded_evaluation(task):
            async with semaphore:
                return await task
        
        bounded_tasks = [bounded_evaluation(task) for task in tasks]
        results = await tqdm.gather(*bounded_tasks, desc="Running evaluations")
        
        # Process results
        self.results = [r for r in results if r is not None]
        df_results = pd.DataFrame([asdict(r) for r in self.results])
        
        # Log results to W&B
        self._log_to_wandb(df_results)
        
        # Save results locally
        self._save_results(df_results, experiment_name)
        
        logger.info(f"Evaluation completed. Results saved for experiment: {experiment_name}")
        
        return df_results
    
    async def _evaluate_single(self, model: BaseModel, test_case: TestCase) -> Optional[EvaluationResult]:
        """Evaluate a single model on a single test case"""
        try:
            # Generate response
            response = await model.generate(
                test_case.prompt,
                max_tokens=self.config.get('max_tokens', 1000),
                temperature=self.config.get('temperature', 0.7)
            )
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                test_case.expected_output,
                response.text
            )
            
            return EvaluationResult(
                model_name=model.model_name,
                test_case_id=test_case.id,
                category=test_case.category,
                prompt=test_case.prompt,
                response=response.text,
                expected_output=test_case.expected_output,
                latency=response.latency,
                tokens_used=response.tokens_used,
                cost=response.cost,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {model.model_name} on {test_case.id}: {e}")
            return EvaluationResult(
                model_name=model.model_name,
                test_case_id=test_case.id,
                category=test_case.category,
                prompt=test_case.prompt,
                response="",
                expected_output=test_case.expected_output,
                latency=0.0,
                tokens_used=0,
                cost=0.0,
                metrics={
                    'bleu': 0.0,
                    'rouge1': 0.0,
                    'rouge2': 0.0,
                    'rougeL': 0.0,
                    'bert_score': 0.0,
                    'semantic_similarity': 0.0,
                    'length_ratio': 0.0,
                    'length': 0,
                    'char_count': 0
                },
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    def _log_to_wandb(self, df: pd.DataFrame):
        """Log results to Weights & Biases"""
        try:
            # Create summary metrics by model
            model_summary = df.groupby('model_name').agg({
                'latency': ['mean', 'std'],
                'cost': 'sum',
                'tokens_used': 'sum'
            }).round(4)
            
            # Log metric summaries
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                
                # Calculate average metrics
                avg_metrics = {}
                for metric_col in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bert_score', 'semantic_similarity']:
                    metric_values = []
                    for _, row in model_data.iterrows():
                        if isinstance(row['metrics'], dict) and metric_col in row['metrics']:
                            metric_values.append(row['metrics'][metric_col])
                    
                    if metric_values:
                        avg_metrics[f"{model}_{metric_col}_mean"] = np.mean(metric_values)
                        avg_metrics[f"{model}_{metric_col}_std"] = np.std(metric_values)
                
                avg_metrics[f"{model}_latency_mean"] = model_data['latency'].mean()
                avg_metrics[f"{model}_total_cost"] = model_data['cost'].sum()
                avg_metrics[f"{model}_total_tokens"] = model_data['tokens_used'].sum()
                
                wandb.log(avg_metrics)
            
            # Create simplified dataframe for W&B logging
            wb_df = df.copy()
            # Flatten metrics into separate columns
            for _, row in wb_df.iterrows():
                if isinstance(row['metrics'], dict):
                    for metric_name, metric_value in row['metrics'].items():
                        wb_df.loc[_, f'metric_{metric_name}'] = metric_value
            
            # Remove the nested metrics column
            wb_df = wb_df.drop('metrics', axis=1)
            
            # Create and log detailed results table
            wandb.log({"detailed_results": wandb.Table(dataframe=wb_df)})
            
            # Log category-wise performance
            category_summary = df.groupby(['model_name', 'category']).agg({
                'latency': 'mean',
                'cost': 'sum'
            }).reset_index()
            
            wandb.log({"category_performance": wandb.Table(dataframe=category_summary)})
            
        except Exception as e:
            logger.error(f"Error logging to W&B: {e}")
    
    def _save_results(self, df: pd.DataFrame, experiment_name: str):
        """Save results to local files"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Create a flattened version for CSV export
        csv_df = df.copy()
        
        # Flatten metrics into separate columns
        metric_columns = {}
        for _, row in csv_df.iterrows():
            if isinstance(row['metrics'], dict):
                for metric_name, metric_value in row['metrics'].items():
                    if metric_name not in metric_columns:
                        metric_columns[metric_name] = []
                    metric_columns[metric_name].append(metric_value)
                else:
                    for metric_name in metric_columns:
                        metric_columns[metric_name].append(0.0)
        
        # Add metric columns to dataframe
        for metric_name, values in metric_columns.items():
            csv_df[f'metric_{metric_name}'] = values[:len(csv_df)]
        
        # Remove the nested metrics column
        csv_df = csv_df.drop('metrics', axis=1)
        
        # Save detailed results
        csv_df.to_csv(results_dir / f"{experiment_name}_detailed.csv", index=False)
        
        # Save summary
        summary = df.groupby('model_name').agg({
            'latency': ['mean', 'std', 'min', 'max'],
            'cost': 'sum',
            'tokens_used': 'sum'
        }).round(4)
        
        summary.to_csv(results_dir / f"{experiment_name}_summary.csv")
        
        # Save config
        with open(results_dir / f"{experiment_name}_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

# =============================================================================
# Configuration Management
# =============================================================================

class ConfigManager:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'wandb_project': 'llm-evaluation',
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
                    {'name': 'gemini-1.5-flash', 'api_key': '${GOOGLE_API_KEY}'},
                    {'name': 'gemini-1.5-pro', 'api_key': '${GOOGLE_API_KEY}'}
                ],
                'ollama': [
                    {'name': 'llama2', 'base_url': 'http://localhost:11434'}
                ]
            },
            'pricing': {
                'openai': {
                    'gpt-4': {'input': 30.00, 'output': 60.00},
                    'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
                    'gpt-4o': {'input': 3.00, 'output': 10.00},
                    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
                    'gpt-3.5-turbo': {'input': 1.50, 'output': 2.00}
                },
                'anthropic': {
                    'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00},
                    'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00},
                    'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
                    'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00}
                },
                'gemini': {
                    'gemini-pro': {'input': 0.50, 'output': 1.50},
                    'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
                    'gemini-1.5-flash': {'input': 0.075, 'output': 0.30}
                },
                'ollama': {
                    'default': {'input': 0.0, 'output': 0.0}
                }
            },
            'default_pricing': {'input': 1.00, 'output': 3.00}
        }

# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    @staticmethod
    def create_models(config: Dict[str, Any]) -> List[BaseModel]:
        """Create model instances from configuration"""
        models = []
        
        # Extract pricing configuration
        pricing_config = config.get('pricing', {})
        pricing_config['default_pricing'] = config.get('default_pricing', {'input': 1.00, 'output': 3.00})
        
        # OpenAI models
        for model_config in config['models'].get('openai', []):
            api_key = os.getenv('OPENAI_API_KEY') if model_config['api_key'] == '${OPENAI_API_KEY}' else model_config['api_key']
            if api_key:
                models.append(OpenAIModel(model_config['name'], api_key, pricing_config))
            else:
                logger.warning(f"No API key found for OpenAI model {model_config['name']}")
        
        # Anthropic models
        for model_config in config['models'].get('anthropic', []):
            api_key = os.getenv('ANTHROPIC_API_KEY') if model_config['api_key'] == '${ANTHROPIC_API_KEY}' else model_config['api_key']
            if api_key:
                models.append(AnthropicModel(model_config['name'], api_key, pricing_config))
            else:
                logger.warning(f"No API key found for Anthropic model {model_config['name']}")
        
        # Google Gemini models
        for model_config in config['models'].get('gemini', []):
            api_key = os.getenv('GOOGLE_API_KEY') if model_config['api_key'] == '${GOOGLE_API_KEY}' else model_config['api_key']
            if api_key:
                models.append(GeminiModel(model_config['name'], api_key, pricing_config))
            else:
                logger.warning(f"No API key found for Gemini model {model_config['name']}")
        
        # Ollama models
        for model_config in config['models'].get('ollama', []):
            models.append(OllamaModel(
                model_config['name'],
                model_config.get('base_url', 'http://localhost:11434'),
                pricing_config
            ))
        
        return models

# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Framework")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--test-suite', type=str, help='Path to test suite file')
    parser.add_argument('--experiment-name', type=str, help='Name for the experiment')
    parser.add_argument('--create-sample', action='store_true', help='Create sample test suite')
    parser.add_argument('--create-config', action='store_true', help='Create default config file')
    
    args = parser.parse_args()
    
    # Create sample files if requested
    if args.create_config:
        config = ConfigManager.create_default_config()
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Created config.yaml with default configuration")
        return
    
    if args.create_sample:
        test_suite = TestSuite.create_sample_suite()
        sample_data = [asdict(tc) for tc in test_suite.test_cases]
        with open('sample_test_suite.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
        print("Created sample_test_suite.json with sample test cases")
        return
    
    # Load configuration
    if args.config:
        config = ConfigManager.load_config(args.config)
    else:
        config = ConfigManager.create_default_config()
    
    # Load test suite
    if args.test_suite:
        test_suite = TestSuite.from_file(args.test_suite)
    else:
        test_suite = TestSuite.create_sample_suite()
        print("Using sample test suite. Use --create-sample to create a file.")
    
    # Create models
    models = ModelFactory.create_models(config)
    if not models:
        print("No models configured. Please check your configuration and API keys.")
        return
    
    print(f"Loaded {len(models)} models: {[m.model_name for m in models]}")
    print(f"Loaded {len(test_suite.test_cases)} test cases")
    
    # Run evaluation
    evaluator = ModelEvaluator(models, test_suite, config)
    
    async def run_async():
        df_results = await evaluator.run_evaluation(args.experiment_name)
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        summary = df_results.groupby('model_name').agg({
            'latency': 'mean',
            'cost': 'sum',
            'tokens_used': 'sum'
        }).round(4)
        print(summary)
        
        print(f"\nDetailed results saved to results/ directory")
        print(f"View results in W&B: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
    
    asyncio.run(run_async())

if __name__ == "__main__":
    main()