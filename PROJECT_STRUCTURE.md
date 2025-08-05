# Project Structure

This document outlines the organization of the LLM Evaluation Framework project.

## 📁 Directory Structure

```
llm_evaluator/
├── README.md                   # Main project documentation
├── LICENSE                     # MIT license
├── CHANGELOG.md               # Version history and changes
├── CONTRIBUTING.md            # Contribution guidelines
├── pyproject.toml             # Modern Python packaging configuration
├── setup.py                   # Legacy Python packaging (for compatibility)
├── requirements.txt           # Python dependencies
├── Makefile                   # Development automation scripts
├── .gitignore                 # Git ignore rules
├── .env.example              # Environment variables template
├── config.yaml               # Main configuration file
├── sample_test_suite.json    # Example test cases
├── main.py                   # Legacy CLI entry point
├── cli.py                    # Modern CLI interface
│
├── src/                      # Main source code package
│   ├── __init__.py          # Package initialization
│   ├── models.py            # Data models (TestCase, ModelResponse, etc.)
│   ├── llm_models.py        # LLM provider implementations
│   ├── config.py            # Configuration management
│   ├── metrics.py           # Evaluation metrics
│   ├── cache.py             # Response caching system
│   ├── utils.py             # Utilities and logging
│   └── exceptions.py        # Custom exception classes
│
├── tests/                   # Test suite
│   └── test_framework.py   # Unit tests
│
├── docs/                    # Documentation (empty, for future use)
├── results/                 # Evaluation results (gitignored)
└── wandb/                   # Weights & Biases logs (gitignored)
```

## 🧩 Module Organization

### Core Modules

- **`src/models.py`**: Data classes for the framework
  - `ModelResponse`: API response wrapper
  - `TestCase`: Individual test case
  - `EvaluationResult`: Complete evaluation result

- **`src/llm_models.py`**: LLM provider implementations
  - `BaseModel`: Abstract base class
  - `OpenAIModel`: OpenAI API integration
  - `AnthropicModel`: Anthropic Claude integration
  - `GeminiModel`: Google Gemini integration
  - `OllamaModel`: Local Ollama integration

- **`src/metrics.py`**: Evaluation metrics
  - `BLEUCalculator`: BLEU score computation
  - `ROUGECalculator`: ROUGE score computation
  - `BERTScoreCalculator`: BERTScore computation
  - `SemanticSimilarityCalculator`: Sentence transformer similarity
  - `CategorySpecificMetrics`: Task-specific evaluation

### Support Modules

- **`src/config.py`**: Configuration management
  - `ConfigManager`: Load and validate configurations
  - `ConfigValidator`: Validate configuration structure

- **`src/cache.py`**: Caching system
  - `ResultsCache`: Cache API responses and metrics

- **`src/utils.py`**: Utilities
  - `PerformanceMonitor`: Track performance metrics
  - `RateLimiter`: API rate limiting
  - Logging setup and formatting

- **`src/exceptions.py`**: Custom exceptions
  - Framework-specific error types

## 🔧 Configuration Files

### Main Configuration (`config.yaml`)
- Model provider settings
- API key references
- Pricing information
- Evaluation parameters

### Development Configuration (`pyproject.toml`)
- Package metadata
- Dependencies
- Development tools configuration
- Build system settings

### Environment Variables (`.env`)
- API keys and secrets
- Workspace-specific settings

## 📊 Data Flow

1. **Configuration Loading**: `ConfigManager` loads and validates `config.yaml`
2. **Model Creation**: `ModelFactory` creates provider instances
3. **Test Loading**: `TestSuite` loads test cases from JSON/CSV
4. **Evaluation**: `ModelEvaluator` runs async evaluation
5. **Metrics**: Individual calculators compute scores
6. **Results**: Output to CSV files and W&B dashboard

## 🎯 Entry Points

### CLI Interface (`cli.py`)
Modern Click-based interface with commands:
- `init-config`: Create default configuration
- `init-tests`: Create sample test suite
- `run`: Execute evaluation
- `validate`: Validate configuration
- `results`: Display results summary

### Legacy Interface (`main.py`)
Original argparse-based interface for backward compatibility

## 🧪 Testing Strategy

### Unit Tests (`tests/test_framework.py`)
- Configuration validation
- Model initialization
- Metric calculations
- Error handling

### Integration Tests (Future)
- End-to-end evaluation workflows
- API provider integration
- Performance benchmarks

## 📈 Monitoring & Logging

### Performance Monitoring
- API response times
- Token usage tracking
- Cost calculations
- Error rates

### Logging Levels
- `DEBUG`: Detailed execution flow
- `INFO`: Key operations and results
- `WARNING`: Non-fatal issues
- `ERROR`: Failures and exceptions

## 🔄 Development Workflow

### Code Style
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Automation (`Makefile`)
- `make format`: Format code
- `make lint`: Run linting
- `make test`: Execute tests
- `make install-dev`: Development setup

## 📦 Packaging

### Distribution
- **PyPI**: Standard Python package
- **GitHub**: Source code repository
- **Docker**: Containerized deployment (future)

### Dependencies
- **Core**: Runtime dependencies
- **Dev**: Development tools
- **Optional**: Provider-specific packages

## 🔒 Security

### API Keys
- Environment variable storage
- `.env` file template
- Gitignored sensitive files

### Data Privacy
- No API responses stored in git
- Configurable cache directory
- Optional result encryption (future)

## 🚀 Deployment

### Local Development
1. Clone repository
2. Install dependencies
3. Configure environment
4. Run evaluations

### Production
- Environment-specific configurations
- Secrets management
- Monitoring integration
- Result persistence

---

*This structure supports scalability, maintainability, and ease of contribution while following Python best practices.*
