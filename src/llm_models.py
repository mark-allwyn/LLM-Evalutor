"""
Base model interface and implementations for different LLM providers.
"""
import time
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import aiohttp
import openai
from openai import AsyncOpenAI
import anthropic
import google.generativeai as genai

from .models import ModelResponse

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all LLM models."""
    
    def __init__(self, model_name: str, pricing_config: Dict[str, Any] = None, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self.pricing_config = pricing_config or {}
        self._cost_per_token = self.get_cost_per_token()
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response to the given prompt."""
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> Dict[str, float]:
        """Get the cost per token for this model."""
        pass
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost based on token usage."""
        input_cost = input_tokens / 1_000_000 * self._cost_per_token.get('input', 0)
        output_cost = output_tokens / 1_000_000 * self._cost_per_token.get('output', 0)
        return input_cost + output_cost


class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
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
        openai_pricing = self.pricing_config.get('openai', {})
        model_pricing = openai_pricing.get(self.model_name, {})
        
        if model_pricing:
            return model_pricing
        
        default_pricing = self.pricing_config.get('default_pricing', {'input': 1.00, 'output': 3.00})
        logger.warning(f"No pricing found for {self.model_name}, using default: {default_pricing}")
        return default_pricing


class AnthropicModel(BaseModel):
    """Anthropic Claude model implementation."""
    
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
        anthropic_pricing = self.pricing_config.get('anthropic', {})
        model_pricing = anthropic_pricing.get(self.model_name, {})
        
        if model_pricing:
            return model_pricing
        
        default_pricing = self.pricing_config.get('default_pricing', {'input': 3.00, 'output': 15.00})
        logger.warning(f"No pricing found for {self.model_name}, using default: {default_pricing}")
        return default_pricing


class GeminiModel(BaseModel):
    """Google Gemini model implementation."""
    
    def __init__(self, model_name: str, api_key: str, pricing_config: Dict[str, Any] = None, **kwargs):
        super().__init__(model_name, pricing_config, **kwargs)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()
        try:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
            )
            
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(prompt, generation_config=generation_config)
            )
            
            latency = time.time() - start_time
            response_text = response.text if response.text else ""
            
            tokens_used = None
            cost = None
            
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
        gemini_pricing = self.pricing_config.get('gemini', {})
        model_pricing = gemini_pricing.get(self.model_name, {})
        
        if model_pricing:
            return model_pricing
        
        default_pricing = self.pricing_config.get('default_pricing', {'input': 0.50, 'output': 1.50})
        logger.warning(f"No pricing found for {self.model_name}, using default: {default_pricing}")
        return default_pricing


class OllamaModel(BaseModel):
    """Ollama local model implementation."""
    
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
                        cost=0.0,
                        metadata={
                            'eval_count': result.get('eval_count'),
                            'prompt_eval_count': result.get('prompt_eval_count')
                        }
                    )
        except Exception as e:
            logger.error(f"Ollama API error for {self.model_name}: {e}")
            raise
    
    def get_cost_per_token(self) -> Dict[str, float]:
        ollama_pricing = self.pricing_config.get('ollama', {})
        model_pricing = ollama_pricing.get(self.model_name)
        
        if model_pricing:
            return model_pricing
        
        return ollama_pricing.get('default', {'input': 0.0, 'output': 0.0})
