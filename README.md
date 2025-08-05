# LLM Evaluation Framework

A comprehensive framework for evaluating and comparing Large Language Models across multiple providers (OpenAI, Anthropic, Google Gemini, Ollama) with automated metrics and Weights & Biases integration.

## 🚀 Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google Gemini, Ollama
- **Comprehensive Metrics**: BLEU, ROUGE, BERTScore, Semantic Similarity
- **Cost Tracking**: Automatic cost calculation with up-to-date pricing
- **Async Evaluation**: Concurrent API calls for faster evaluation
- **W&B Integration**: Automatic logging to Weights & Biases
- **Flexible Test Suites**: JSON, JSONL, CSV support
- **Category-based Analysis**: Organize tests by category (math, reasoning, etc.)

## 🛠 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd llm_evaluator

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## 📊 Quick Start

1. **Create configuration:**
```bash
python main.py --create-config
```

2. **Create sample test suite:**
```bash
python main.py --create-sample
```

3. **Run evaluation:**
```bash
python main.py --config config.yaml --test-suite sample_test_suite.json
```

## 🔧 Configuration

Edit `config.yaml` to:
- Add/remove models
- Update pricing information
- Configure W&B settings
- Set evaluation parameters

## 📝 Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
WANDB_API_KEY=your_wandb_key
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=your_wandb_project
```

## 📈 Results

Results are saved to:
- `results/` directory (CSV files)
- Weights & Biases dashboard
- Detailed metrics and summaries

## 🧪 Test Suite Format

```json
[
  {
    "id": "test_1",
    "prompt": "Your test prompt",
    "expected_output": "Expected response (optional)",
    "category": "category_name",
    "metadata": {"difficulty": "medium"}
  }
]
```

## 🎯 Evaluation Metrics

The framework uses multiple complementary metrics to provide a comprehensive evaluation of model performance:

### 📝 Content Quality Metrics

#### **BLEU Score**
- **What it measures**: N-gram overlap between generated text and reference answer
- **Range**: 0.0 to 1.0 (higher is better)
- **Best for**: Tasks with precise, factual answers (math, translations, code)
- **Limitations**: Doesn't account for semantic meaning; sensitive to exact word matches
- **Example**: Perfect match = 1.0, no overlap = 0.0

#### **ROUGE Scores**
- **ROUGE-1**: Unigram (single word) overlap
- **ROUGE-2**: Bigram (two consecutive words) overlap  
- **ROUGE-L**: Longest Common Subsequence between texts
- **Range**: 0.0 to 1.0 (higher is better)
- **Best for**: Summarization tasks, content coverage evaluation
- **Advantage**: Captures recall-oriented similarity (how much reference content is covered)

#### **BERTScore**
- **What it measures**: Contextual embedding similarity using BERT
- **Range**: Typically 0.0 to 1.0 (higher is better)
- **Best for**: Tasks where semantic meaning matters more than exact wording
- **Advantage**: Understands synonyms and paraphrases (e.g., "big" vs "large")
- **Use case**: Creative writing, explanations, paraphrasing tasks

#### **Semantic Similarity**
- **What it measures**: Sentence-level semantic similarity using sentence transformers
- **Range**: -1.0 to 1.0 (higher is better)
- **Best for**: Overall meaning comparison, intent matching
- **Model**: Uses `all-MiniLM-L6-v2` for fast, accurate embeddings
- **Advantage**: Captures high-level meaning even with different word choices

### ⚡ Performance Metrics

#### **Latency**
- **What it measures**: Response time from API call to completion
- **Units**: Seconds
- **Use case**: Compare model speed, identify bottlenecks

#### **Token Usage**
- **What it measures**: Total tokens (input + output) consumed
- **Use case**: Efficiency comparison, usage monitoring

#### **Cost**
- **What it measures**: Estimated cost based on current provider pricing
- **Units**: USD
- **Use case**: Budget planning, cost-effectiveness analysis

### 📊 Metric Interpretation Guide

| Metric Type | When to Use | Interpretation |
|-------------|-------------|----------------|
| **BLEU** | Factual QA, Math, Code | > 0.7 = Excellent, 0.3-0.7 = Good, < 0.3 = Poor |
| **ROUGE-L** | Summarization, Coverage | > 0.5 = Good coverage, < 0.3 = Missing content |
| **BERTScore** | Semantic tasks | > 0.85 = High similarity, 0.7-0.85 = Moderate, < 0.7 = Low |
| **Semantic Sim** | Intent matching | > 0.8 = Very similar, 0.5-0.8 = Related, < 0.5 = Different |

### 🔄 Metric Selection by Task Type

- **Mathematical Problems**: BLEU + Exact Match
- **Creative Writing**: BERTScore + Semantic Similarity  
- **Summarization**: ROUGE scores + BERTScore
- **Code Generation**: BLEU + Syntax validation
- **Question Answering**: All metrics for comprehensive evaluation
- **Open-ended Tasks**: BERTScore + Semantic Similarity (no expected output needed)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details
