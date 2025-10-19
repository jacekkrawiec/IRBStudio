"""
Main entry point and high-level API for IRBStudio.

This module provides a simplified, user-friendly interface to run complete
portfolio analyses without needing to manually instantiate multiple classes.

Example usage:
    >>> from irbstudio import run_analysis
    >>> 
    >>> results = run_analysis(
    ...     config_path='config.yaml',
    ...     portfolio_path='portfolio.csv',
    ...     output_dir='results'
    ... )
    >>> 
    >>> print(results['summary'])
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import time

from .config.schema import Config, Scenario
from .data.loader import load_portfolio
from .simulation.portfolio_simulator import PortfolioSimulator
from .engine.integrated_analysis import IntegratedAnalysis
from .engine.mortgage.airb_calculator import AIRBMortgageCalculator
from .engine.mortgage.sa_calculator import SAMortgageCalculator
from .utils.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load and validate configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
        pydantic.ValidationError: If config validation fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Validate with Pydantic
    config = Config(**raw_config)
    logger.info(f"Configuration loaded successfully with {len(config.scenarios)} scenario(s)")
    
    return config


def run_analysis(
    config_path: Union[str, Path],
    portfolio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    calculators: Optional[List[str]] = None,
    n_iterations: int = 100,
    memory_efficient: bool = True,
    store_full_portfolio: bool = False,
    random_seed: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Run a complete IRBStudio analysis from configuration and portfolio data.
    
    This is the main high-level API that orchestrates the entire workflow:
    1. Load and validate configuration
    2. Load and validate portfolio data
    3. Create portfolio simulators for each scenario
    4. Create RWA calculators
    5. Run Monte Carlo simulations
    6. Calculate RWA for each scenario
    7. Generate comparative analysis
    8. Export results (if output_dir provided)
    
    Args:
        config_path: Path to YAML configuration file
        portfolio_path: Path to portfolio CSV or Parquet file
        output_dir: Optional directory to save results. If None, results are only returned.
        calculators: List of calculator types to use. Options: ['AIRB', 'SA'].
                    If None, uses both AIRB and SA.
        n_iterations: Number of Monte Carlo iterations per scenario (default: 100)
        memory_efficient: If True, process iterations one at a time to save memory
        store_full_portfolio: If True, store complete portfolio DataFrames in results
        random_seed: Random seed for reproducibility
        progress_callback: Optional callback function(step: str, progress: float)
                          for progress tracking
    
    Returns:
        Dictionary containing:
            - 'config': The validated configuration object
            - 'portfolio_stats': Basic statistics about the loaded portfolio
            - 'scenarios': Dict of scenario results, keyed by scenario name
            - 'comparisons': Comparative analysis between scenarios
            - 'execution_time': Total execution time in seconds
            - 'output_files': List of generated output files (if output_dir provided)
    
    Example:
        >>> results = run_analysis(
        ...     config_path='config.yaml',
        ...     portfolio_path='my_portfolio.csv',
        ...     output_dir='analysis_results',
        ...     calculators=['AIRB', 'SA'],
        ...     n_iterations=1000,
        ...     random_seed=42
        ... )
        >>> 
        >>> # Access results
        >>> print(f"Baseline AIRB RWA: {results['scenarios']['Baseline']['AIRB']['mean_rwa']:,.0f}")
    """
    start_time = time.time()
    
    def report_progress(step: str, progress: float):
        """Helper to report progress"""
        logger.info(f"[{progress:.0%}] {step}")
        if progress_callback:
            progress_callback(step, progress)
    
    # Step 1: Load configuration (5%)
    report_progress("Loading configuration", 0.05)
    config = load_config(config_path)
    
    # Step 2: Load portfolio data (10%)
    report_progress("Loading portfolio data", 0.10)
    portfolio_df = load_portfolio(portfolio_path, config.column_mapping)
    
    # Calculate portfolio statistics
    portfolio_stats = {
        'n_loans': len(portfolio_df),
        'n_unique_ids': portfolio_df['loan_id'].nunique() if 'loan_id' in portfolio_df.columns else None,
        'total_exposure': portfolio_df['exposure'].sum() if 'exposure' in portfolio_df.columns else None,
        'columns': list(portfolio_df.columns)
    }
    
    # Step 3: Initialize calculators (15%)
    report_progress("Initializing RWA calculators", 0.15)
    
    if calculators is None:
        calculators = ['AIRB', 'SA']
    
    calculator_instances = {}
    
    if 'AIRB' in calculators:
        calculator_instances['AIRB'] = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': config.regulatory.asset_correlation,
                'confidence_level': config.regulatory.confidence_level
            }
        )
        logger.info("AIRB calculator initialized")
    
    if 'SA' in calculators:
        calculator_instances['SA'] = SAMortgageCalculator(
            regulatory_params={
                'confidence_level': config.regulatory.confidence_level
            }
        )
        logger.info("SA calculator initialized")
    
    # Step 4: Create IntegratedAnalysis framework (20%)
    report_progress("Setting up integrated analysis framework", 0.20)
    
    analysis = IntegratedAnalysis(
        calculators=calculator_instances,
        date_column='date',  # We use 'date' as the standardized column name
        pd_column='simulated_pd',
        target_pd_column='pd'
    )
    
    # Step 5: Create simulators and add scenarios (25%)
    report_progress("Preparing portfolio simulators", 0.25)
    
    # Derive score-to-rating bounds from portfolio data
    # Group by rating and get min/max scores for each rating
    if 'rating' in portfolio_df.columns and 'score' in portfolio_df.columns:
        rating_score_bounds = {}
        for rating in sorted(portfolio_df['rating'].unique()):
            rating_scores = portfolio_df[portfolio_df['rating'] == rating]['score']
            if len(rating_scores) > 0:
                rating_score_bounds[rating] = (
                    float(rating_scores.min()),
                    float(rating_scores.max())
                )
        logger.info(f"Derived score-to-rating bounds from portfolio: {rating_score_bounds}")
    else:
        # Fallback if no rating or score columns
        rating_score_bounds = {}
        logger.warning("Could not derive score-to-rating bounds: missing 'rating' or 'score' columns")
    
    # Create simulators for each scenario
    for idx, scenario in enumerate(config.scenarios):
        scenario_progress = 0.25 + (idx / len(config.scenarios)) * 0.15
        report_progress(f"Setting up scenario: {scenario.name}", scenario_progress)
        
        # Create simulator for this scenario
        # The portfolio has standardized column names after load_portfolio()
        simulator = PortfolioSimulator(
            portfolio_df=portfolio_df.copy(),
            score_to_rating_bounds=rating_score_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            exposure_col='exposure' if 'exposure' in portfolio_df.columns else None,
            target_auc=scenario.pd_auc if hasattr(scenario, 'pd_auc') else None,
            asset_correlation=config.regulatory.asset_correlation,
            random_seed=random_seed
        )
        
        # Add scenario to analysis
        analysis.add_scenario(
            name=scenario.name,
            simulator=simulator,
            n_iterations=n_iterations
        )
        
        logger.info(f"Scenario '{scenario.name}' configured with {n_iterations} iterations")
    
    # Step 6: Run simulations (40% - 85%)
    scenario_results = {}
    total_scenarios = len(config.scenarios)
    
    for idx, scenario in enumerate(config.scenarios):
        base_progress = 0.40 + (idx / total_scenarios) * 0.45
        end_progress = 0.40 + ((idx + 1) / total_scenarios) * 0.45
        
        report_progress(f"Running Monte Carlo simulation: {scenario.name}", base_progress)
        
        # Run this scenario
        results = analysis.run_scenario(
            scenario_name=scenario.name,
            calculator_names=list(calculator_instances.keys()),
            memory_efficient=memory_efficient,
            store_full_portfolio=store_full_portfolio
        )
        
        # Collect statistics for each calculator
        scenario_calc_results = {}
        
        for calc_name in calculator_instances.keys():
            stats = analysis.get_summary_stats(scenario.name, calc_name)
            scenario_calc_results[calc_name] = stats
            
            logger.info(
                f"Scenario '{scenario.name}' - {calc_name}: "
                f"Mean RWA = {stats.get('mean', 0):,.0f}, "
                f"Std Dev = {stats.get('std', 0):,.0f}"
            )
        
        scenario_results[scenario.name] = scenario_calc_results
        report_progress(f"Completed scenario: {scenario.name}", end_progress)
    
    # Step 7: Generate comparative analysis (90%)
    report_progress("Generating comparative analysis", 0.90)
    
    comparisons = {}
    
    # If there are multiple scenarios, create comparisons
    if len(config.scenarios) > 1:
        baseline_name = config.scenarios[0].name
        
        for scenario in config.scenarios[1:]:
            for calc_name in calculator_instances.keys():
                comparison_key = f"{scenario.name}_vs_{baseline_name}_{calc_name}"
                
                baseline_stats = scenario_results[baseline_name][calc_name]
                scenario_stats = scenario_results[scenario.name][calc_name]
                
                comparisons[comparison_key] = {
                    'baseline_mean': baseline_stats.get('mean', 0),
                    'scenario_mean': scenario_stats.get('mean', 0),
                    'absolute_difference': scenario_stats.get('mean', 0) - baseline_stats.get('mean', 0),
                    'relative_difference': (
                        (scenario_stats.get('mean', 0) - baseline_stats.get('mean', 0)) / 
                        baseline_stats.get('mean', 1)
                    ) if baseline_stats.get('mean', 0) != 0 else 0
                }
    
    # Step 8: Export results if output directory provided (95%)
    output_files = []
    
    if output_dir:
        report_progress("Exporting results", 0.95)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export summary to CSV
        summary_file = output_path / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        summary_data = []
        for scenario_name, calc_results in scenario_results.items():
            for calc_name, stats in calc_results.items():
                summary_data.append({
                    'scenario': scenario_name,
                    'calculator': calc_name,
                    **stats
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        output_files.append(str(summary_file))
        
        logger.info(f"Summary exported to: {summary_file}")
        
        # Export comparisons if available
        if comparisons:
            comp_file = output_path / f"comparisons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            comp_df = pd.DataFrame([
                {'comparison': k, **v} for k, v in comparisons.items()
            ])
            comp_df.to_csv(comp_file, index=False)
            output_files.append(str(comp_file))
            
            logger.info(f"Comparisons exported to: {comp_file}")
    
    # Step 9: Finalize (100%)
    execution_time = time.time() - start_time
    report_progress("Analysis complete", 1.0)
    
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    
    # Compile final results
    return {
        'config': config,
        'portfolio_stats': portfolio_stats,
        'scenarios': scenario_results,
        'comparisons': comparisons,
        'execution_time': execution_time,
        'output_files': output_files,
        'analysis_timestamp': datetime.now().isoformat()
    }


def run_scenario_comparison(
    config_path: Union[str, Path],
    portfolio_path: Union[str, Path],
    baseline_scenario: str,
    comparison_scenarios: List[str],
    calculator: str = 'AIRB',
    n_iterations: int = 100,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Run a focused comparison between a baseline scenario and alternative scenarios.
    
    This is a specialized version of run_analysis() that focuses on comparing
    specific scenarios against a baseline, with detailed delta analysis.
    
    Args:
        config_path: Path to YAML configuration file
        portfolio_path: Path to portfolio CSV or Parquet file
        baseline_scenario: Name of the baseline scenario
        comparison_scenarios: List of scenario names to compare against baseline
        calculator: Calculator to use ('AIRB' or 'SA')
        n_iterations: Number of Monte Carlo iterations per scenario
        output_dir: Optional directory to save comparison results
    
    Returns:
        Dictionary with detailed comparison metrics including:
            - Mean RWA differences
            - Standard deviation comparisons
            - Percentile comparisons
            - Capital impact estimates
    """
    logger.info(f"Running scenario comparison: {comparison_scenarios} vs {baseline_scenario}")
    
    # Run full analysis
    results = run_analysis(
        config_path=config_path,
        portfolio_path=portfolio_path,
        output_dir=output_dir,
        calculators=[calculator],
        n_iterations=n_iterations
    )
    
    # Extract baseline results
    baseline_stats = results['scenarios'].get(baseline_scenario, {}).get(calculator, {})
    
    if not baseline_stats:
        raise ValueError(f"Baseline scenario '{baseline_scenario}' not found or has no {calculator} results")
    
    # Build detailed comparisons
    detailed_comparisons = {}
    
    for comp_scenario in comparison_scenarios:
        comp_stats = results['scenarios'].get(comp_scenario, {}).get(calculator, {})
        
        if not comp_stats:
            logger.warning(f"Comparison scenario '{comp_scenario}' not found, skipping")
            continue
        
        detailed_comparisons[comp_scenario] = {
            'baseline_mean_rwa': baseline_stats.get('mean', 0),
            'scenario_mean_rwa': comp_stats.get('mean', 0),
            'absolute_rwa_change': comp_stats.get('mean', 0) - baseline_stats.get('mean', 0),
            'relative_rwa_change': (
                (comp_stats.get('mean', 0) - baseline_stats.get('mean', 0)) / baseline_stats.get('mean', 1)
            ) if baseline_stats.get('mean', 0) != 0 else 0,
            'baseline_std': baseline_stats.get('std', 0),
            'scenario_std': comp_stats.get('std', 0),
            'uncertainty_change': comp_stats.get('std', 0) - baseline_stats.get('std', 0),
        }
        
        # Add percentile comparisons if available
        for pct in ['p5', 'p25', 'p50', 'p75', 'p95']:
            if pct in baseline_stats and pct in comp_stats:
                detailed_comparisons[comp_scenario][f'{pct}_baseline'] = baseline_stats[pct]
                detailed_comparisons[comp_scenario][f'{pct}_scenario'] = comp_stats[pct]
                detailed_comparisons[comp_scenario][f'{pct}_difference'] = comp_stats[pct] - baseline_stats[pct]
    
    return {
        'baseline': baseline_scenario,
        'calculator': calculator,
        'comparisons': detailed_comparisons,
        'execution_time': results['execution_time'],
        'portfolio_stats': results['portfolio_stats']
    }
