"""
Caching utilities for LLM evaluation framework.
"""
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ResultsCache:
    """Cache for evaluation results to avoid re-running expensive operations."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "responses").mkdir(exist_ok=True)
        (self.cache_dir / "metrics").mkdir(exist_ok=True)
    
    def _get_cache_key(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate a cache key for a model response."""
        # Include model name, prompt, and generation parameters in the hash
        cache_data = {
            'model': model_name,
            'prompt': prompt,
            'max_tokens': kwargs.get('max_tokens', 1000),
            'temperature': kwargs.get('temperature', 0.7)
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_response(self, model_name: str, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        cache_key = self._get_cache_key(model_name, prompt, **kwargs)
        cache_file = self.cache_dir / "responses" / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_response = pickle.load(f)
                logger.debug(f"Cache hit for {model_name}: {cache_key}")
                return cached_response
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def save_response(self, model_name: str, prompt: str, response_data: Dict[str, Any], **kwargs):
        """Save response to cache."""
        cache_key = self._get_cache_key(model_name, prompt, **kwargs)
        cache_file = self.cache_dir / "responses" / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(response_data, f)
            logger.debug(f"Cached response for {model_name}: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            (self.cache_dir / "responses").mkdir(exist_ok=True)
            (self.cache_dir / "metrics").mkdir(exist_ok=True)
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        response_cache = self.cache_dir / "responses"
        metrics_cache = self.cache_dir / "metrics"
        
        return {
            'response_files': len(list(response_cache.glob("*.pkl"))) if response_cache.exists() else 0,
            'metrics_files': len(list(metrics_cache.glob("*.pkl"))) if metrics_cache.exists() else 0,
        }
