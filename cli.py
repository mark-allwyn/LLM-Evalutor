#!/usr/bin/env python3
"""
Improved CLI interface for the LLM Evaluation Framework.
"""
import click
import asyncio
import json
import yaml
from pathlib import Path
from typing import Optional

from src import (
    ConfigManager, 
    setup_logging,
    LLMEvaluatorError,
    ModelConfigError
)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-level', default='INFO', help='Set log level (DEBUG, INFO, WARNING, ERROR)')
@click.pass_context
def cli(ctx, verbose, log_level):
    """LLM Evaluation Framework CLI."""
    ctx.ensure_object(dict)
    
    if verbose:
        log_level = 'DEBUG'
    
    ctx.obj['logger'] = setup_logging(log_level)


@cli.command()
@click.option('--output', '-o', default='config.yaml', help='Output configuration file')
def init_config(output):
    """Create a default configuration file."""
    try:
        config = ConfigManager.create_default_config()
        
        output_path = Path(output)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        click.echo(f"✅ Configuration created: {output_path}")
        click.echo("📝 Edit the config file and set your API keys in .env")
        
    except Exception as e:
        click.echo(f"❌ Error creating config: {e}")
        raise click.Abort()


@cli.command()
@click.option('--output', '-o', default='sample_test_suite.json', help='Output test suite file')
@click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'jsonl', 'csv']))
def init_tests(output, output_format):
    """Create a sample test suite."""
    from src.test_suite import TestSuite  # This would need to be implemented
    
    try:
        test_suite = TestSuite.create_sample_suite()
        output_path = Path(output)
        
        if output_format == 'json':
            test_suite.save_as_json(output_path)
        elif output_format == 'jsonl':
            test_suite.save_as_jsonl(output_path)
        elif output_format == 'csv':
            test_suite.save_as_csv(output_path)
        
        click.echo(f"✅ Sample test suite created: {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Error creating test suite: {e}")
        raise click.Abort()


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Configuration file')
@click.option('--test-suite', '-t', required=True, help='Test suite file')
@click.option('--experiment-name', '-n', help='Experiment name')
@click.option('--dry-run', is_flag=True, help='Validate configuration without running')
@click.option('--models', help='Comma-separated list of models to run (optional)')
@click.option('--categories', help='Comma-separated list of categories to test (optional)')
@click.pass_context
def run(ctx, config, test_suite, experiment_name, dry_run, models, categories):
    """Run the evaluation."""
    logger = ctx.obj['logger']
    
    try:
        # Load and validate configuration
        config_data = ConfigManager.load_config(config)
        logger.info(f"✅ Configuration loaded from {config}")
        
        if dry_run:
            click.echo("✅ Configuration is valid!")
            click.echo(f"📊 Models configured: {sum(len(m) for m in config_data['models'].values())}")
            return
        
        # Run the actual evaluation
        asyncio.run(_run_evaluation(config_data, test_suite, experiment_name, models, categories, logger))
        
    except ModelConfigError as e:
        click.echo(f"❌ Configuration error: {e}")
        raise click.Abort()
    except FileNotFoundError as e:
        click.echo(f"❌ File not found: {e}")
        raise click.Abort()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        click.echo(f"❌ Error: {e}")
        raise click.Abort()


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Configuration file to validate')
def validate(config):
    """Validate configuration file."""
    try:
        config_data = ConfigManager.load_config(config)
        click.echo("✅ Configuration is valid!")
        
        # Show summary
        models_count = sum(len(models) for models in config_data['models'].values())
        click.echo(f"📊 Total models configured: {models_count}")
        
        for provider, models in config_data['models'].items():
            click.echo(f"  • {provider}: {len(models)} models")
            
    except ModelConfigError as e:
        click.echo(f"❌ Configuration error: {e}")
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise click.Abort()


@cli.command()
@click.option('--results-dir', default='results', help='Results directory')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json', 'csv']))
def results(results_dir, output_format):
    """Show evaluation results summary."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        click.echo(f"❌ Results directory not found: {results_dir}")
        raise click.Abort()
    
    # Find the most recent results
    summary_files = list(results_path.glob("*_summary.csv"))
    
    if not summary_files:
        click.echo(f"❌ No summary files found in {results_dir}")
        raise click.Abort()
    
    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
    
    try:
        import pandas as pd
        df = pd.read_csv(latest_summary)
        
        if output_format == 'table':
            click.echo(f"📊 Latest results from: {latest_summary.name}")
            click.echo(df.to_string())
        elif output_format == 'json':
            click.echo(df.to_json(indent=2))
        elif output_format == 'csv':
            click.echo(df.to_csv())
            
    except Exception as e:
        click.echo(f"❌ Error reading results: {e}")
        raise click.Abort()


async def _run_evaluation(config_data, test_suite_path, experiment_name, models_filter, categories_filter, logger):
    """Run the actual evaluation (async)."""
    # This would import and use the refactored evaluator classes
    from main import ModelEvaluator, TestSuite, ModelFactory
    
    # Load test suite
    test_suite = TestSuite.from_file(test_suite_path)
    logger.info(f"📝 Loaded {len(test_suite.test_cases)} test cases")
    
    # Filter categories if specified
    if categories_filter:
        categories = [c.strip() for c in categories_filter.split(',')]
        original_count = len(test_suite.test_cases)
        test_suite.test_cases = [tc for tc in test_suite.test_cases if tc.category in categories]
        logger.info(f"🔍 Filtered to {len(test_suite.test_cases)} test cases from {original_count}")
    
    # Create models
    models = ModelFactory.create_models(config_data)
    
    # Filter models if specified
    if models_filter:
        model_names = [m.strip() for m in models_filter.split(',')]
        models = [m for m in models if m.model_name in model_names]
        logger.info(f"🤖 Running with {len(models)} models: {[m.model_name for m in models]}")
    else:
        logger.info(f"🤖 Running with {len(models)} models")
    
    if not models:
        raise LLMEvaluatorError("No models available for evaluation")
    
    # Run evaluation
    evaluator = ModelEvaluator(models, test_suite, config_data)
    results_df = await evaluator.run_evaluation(experiment_name)
    
    logger.info("✅ Evaluation completed!")
    logger.info(f"📊 Results saved to results/ directory")


if __name__ == '__main__':
    cli()
