# Contributing to LLM Evaluation Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Development Environment

1. **Fork and clone the repository:**
```bash
git clone https://github.com/yourusername/llm-evaluator.git
cd llm-evaluator
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run tests to verify setup:**
```bash
make test
```

## 🔧 Development Workflow

### Code Style
- Use **Black** for code formatting: `make format`
- Use **flake8** for linting: `make lint`
- Use **mypy** for type checking: `make lint`
- Follow PEP 8 style guidelines

### Making Changes

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes:**
   - Write clear, concise code
   - Add type hints
   - Include docstrings for functions/classes
   - Update tests as needed

3. **Test your changes:**
```bash
make test          # Run all tests
make test-cov      # Run with coverage
make lint          # Check code style
```

4. **Commit your changes:**
```bash
git add .
git commit -m "feat: add your feature description"
```

### Commit Message Format
Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding tests
- `chore:` maintenance tasks

## 📝 Guidelines

### Adding New Models
To add support for a new LLM provider:

1. Create a new model class in `src/llm_models.py` inheriting from `BaseModel`
2. Implement required methods: `generate()` and `get_cost_per_token()`
3. Add pricing configuration to default config
4. Update `ModelFactory.create_models()`
5. Add tests in `tests/test_framework.py`
6. Update documentation

### Adding New Metrics
To add a new evaluation metric:

1. Create metric class in `src/metrics.py`
2. Implement calculation logic
3. Add to `AdvancedMetrics.calculate_all_metrics()`
4. Add tests for the metric
5. Update README documentation

### Testing
- Write unit tests for all new functionality
- Use `pytest` for testing framework
- Include both positive and negative test cases
- Mock external API calls in tests
- Aim for >80% test coverage

### Documentation
- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Update CHANGELOG.md

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Environment details:**
   - Python version
   - Operating system
   - Dependency versions

2. **Steps to reproduce:**
   - Clear step-by-step instructions
   - Code snippets if applicable
   - Configuration files (remove API keys!)

3. **Expected vs actual behavior:**
   - What you expected to happen
   - What actually happened
   - Error messages or logs

4. **Additional context:**
   - Screenshots if relevant
   - Related issues or PRs

## 📋 Pull Request Process

1. **Ensure all tests pass:**
```bash
make test
make lint
```

2. **Update documentation:**
   - README if user-facing changes
   - CHANGELOG.md with your changes
   - Code docstrings

3. **Create pull request:**
   - Use clear, descriptive title
   - Fill out PR template
   - Reference related issues
   - Include screenshots if UI changes

4. **Review process:**
   - Address review feedback
   - Keep PR focused and atomic
   - Maintain clean commit history

## 🏗️ Development Setup Commands

```bash
# Setup
make install-dev        # Install in development mode
make setup-hooks       # Install pre-commit hooks

# Development
make format            # Format code with Black
make lint              # Run linting and type checks
make test              # Run tests
make test-cov          # Run tests with coverage

# Validation
make validate-config   # Validate configuration
make example          # Run example evaluation

# Cleanup
make clean            # Remove cache files and build artifacts
```

## 💡 Feature Requests

We welcome feature requests! Please:

1. Check existing issues for duplicates
2. Clearly describe the use case
3. Explain why it would be valuable
4. Consider implementation complexity
5. Be willing to contribute if possible

## 📞 Getting Help

- **GitHub Issues:** For bugs and feature requests
- **Discussions:** For questions and general discussion
- **Email:** For security issues or private concerns

## 🙏 Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for their contributions
- README.md contributors section
- Release notes for significant contributions

Thank you for contributing to making LLM evaluation better for everyone!
