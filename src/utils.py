"""
Utilities for logging, monitoring, and performance tracking.
"""
import logging
import time
import functools
from typing import Any, Callable, Dict
from contextlib import contextmanager
from datetime import datetime

import wandb


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up colored logging."""
    logger = logging.getLogger("llm_evaluator")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


class PerformanceMonitor:
    """Monitor performance metrics during evaluation."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager to measure operation time."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(operation_name, duration)
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary statistics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'total': sum(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        return summary
    
    def log_to_wandb(self, prefix: str = "performance"):
        """Log performance metrics to W&B."""
        summary = self.get_summary()
        for metric_name, stats in summary.items():
            for stat_name, value in stats.items():
                wandb.log({f"{prefix}/{metric_name}/{stat_name}": value})


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger = logging.getLogger("llm_evaluator")
            logger.debug(f"{func.__name__} took {duration:.4f}s")
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger = logging.getLogger("llm_evaluator")
            logger.debug(f"{func.__name__} took {duration:.4f}s")
    
    return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire a rate limit token."""
        import asyncio
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # If we've hit the limit, wait
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                return await self.acquire()
        
        # Record this call
        self.calls.append(now)


def create_experiment_metadata() -> Dict[str, Any]:
    """Create metadata for the current experiment."""
    return {
        'experiment_id': f"llm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'framework_version': '1.0.0',  # Update this as your framework evolves
        'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
    }
