# Changelog

All notable changes to the LLM Evaluation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-05

### Added
- Initial release of LLM Evaluation Framework
- Multi-provider support (OpenAI, Anthropic, Google Gemini, Ollama)
- Comprehensive evaluation metrics (BLEU, ROUGE, BERTScore, Semantic Similarity)
- Async evaluation with concurrent API calls
- Weights & Biases integration
- Cost tracking with up-to-date pricing
- Flexible test suite formats (JSON, JSONL, CSV)
- Category-based analysis
- Configuration validation
- Caching system for API responses
- Advanced CLI interface with Click
- Comprehensive test suite
- Performance monitoring and logging
- Modular architecture for easy extension

### Features
- **Models**: Support for GPT-4o, GPT-3.5-turbo, Claude-3, Gemini-1.5, and local Ollama models
- **Metrics**: Multiple evaluation metrics with category-specific optimizations
- **Configuration**: YAML-based configuration with environment variable support
- **Results**: CSV export and W&B dashboard integration
- **Testing**: Unit tests with pytest and async support
- **Documentation**: Comprehensive README with usage examples

### Technical
- Python 3.8+ support
- Async/await pattern for performance
- Type hints throughout codebase
- Custom exception handling
- Modular package structure
- Development tools (Makefile, setup.py)

## [Unreleased]

### Planned
- Support for more model providers (Cohere, Together AI)
- Custom metric plugins
- Web dashboard interface
- A/B testing capabilities
- Result comparison and benchmarking
- Integration with HuggingFace models
- Fine-tuning evaluation support
