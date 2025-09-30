"""
This module provides a framework for integrated analysis of portfolio simulations and RWA calculations.

The IntegratedAnalysis class connects the Monte Carlo simulation engine with various RWA calculators,
allowing for comprehensive scenario analysis and impact assessment of different model assumptions
on capital requirements.

For large datasets (10M+ rows), use the memory_efficient=True parameter in the run_scenario method
to process iterations one at a time without storing full DataFrames in memory. This approach:

1. Processes one Monte Carlo iteration at a time
2. Immediately calculates RWA for that iteration
3. Stores only the summary metrics, not the full simulation results
4. Uses garbage collection to free memory between iterations

Column mapping parameters can be used to adapt to different column naming conventions:
- date_column: Name of the column containing reporting dates (default: 'reporting_date')
- pd_column: Name of the simulated PD column (default: 'simulated_pd')
- target_pd_column: Name expected by calculators for PD values (default: 'pd')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional, Callable
import time
import copy

from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.utils.logging import get_logger


class IntegratedAnalysis:
    """
    Integrates Monte Carlo simulations with RWA calculators for comprehensive scenario analysis.
    
    This class serves as a bridge between the portfolio simulation engine and RWA calculators,
    enabling the comparison of multiple scenarios and quantification of their impact on capital
    requirements. It supports:
    
    1. Running multiple Monte Carlo simulation scenarios
    2. Processing simulation results through different RWA calculators
    3. Calculating statistical metrics across simulations
    4. Comparing scenarios to quantify capital impact
    
    The class follows a flexible, dictionary-based design that allows for:
    - Adding multiple calculators (AIRB, SA, etc.)
    - Defining various scenarios with different simulation parameters
    - Storing and analyzing simulation results with full distributions
    """
    
    def __init__(self, 
                 calculators: Optional[Dict[str, Any]] = None,
                 date_column: str = 'reporting_date',
                 pd_column: str = 'simulated_pd',
                 target_pd_column: str = 'pd'):
        """
        Initialize an IntegratedAnalysis instance.
        
        Args:
            calculators: Optional dictionary mapping calculator names to calculator instances.
                         If None, an empty dictionary is initialized.
            date_column: Name of the column containing reporting dates.
            pd_column: Name of the column containing simulated PD values.
            target_pd_column: Name of the PD column expected by calculators.
        """
        self.calculators = calculators or {}
        self.scenarios = {}
        self.results = {}
        self.column_mapping = {
            'date': date_column,
            'pd': pd_column,
            'target_pd': target_pd_column
        }
        self.logger = get_logger(__name__)
    
    def add_calculator(self, name: str, calculator: Any) -> None:
        """
        Add an RWA calculator to the analysis.
        
        Args:
            name: A unique identifier for the calculator (e.g., 'AIRB', 'SA')
            calculator: An instance of an RWA calculator with a calculate() method
        """
        if name in self.calculators:
            self.logger.warning(f"Calculator '{name}' already exists and will be overwritten.")
        
        self.calculators[name] = calculator
    
    def add_scenario(self, 
                    name: str, 
                    simulator: PortfolioSimulator, 
                    n_iterations: int = 100, 
                    **scenario_params) -> None:
        """
        Register a simulation scenario with parameters.
        
        Args:
            name: A unique identifier for the scenario
            simulator: A prepared PortfolioSimulator instance
            n_iterations: Number of Monte Carlo iterations to run
            **scenario_params: Additional parameters specific to this scenario
        """
        if name in self.scenarios:
            self.logger.warning(f"Scenario '{name}' already exists and will be overwritten.")
        
        # Store a deep copy of the simulator to prevent cross-scenario contamination
        self.scenarios[name] = {
            'simulator': simulator,  # Don't copy the simulator to avoid memory issues
            'n_iterations': n_iterations,
            'params': scenario_params
        }
    
    def run_scenario(self, 
                    scenario_name: str, 
                    calculator_names: Optional[List[str]] = None,
                    portfolio_filter: Optional[Callable] = None,
                    random_seed: Optional[int] = None,
                    memory_efficient: bool = True,
                    process_all_dates: bool = False) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation and calculate RWA for a scenario.
        
        Args:
            scenario_name: Name of the previously registered scenario to run
            calculator_names: List of calculator names to apply to simulation results.
                             If None, all registered calculators are used.
            portfolio_filter: Optional function to filter/process the simulation results
                             before passing to calculators
            random_seed: Optional random seed for reproducibility
            memory_efficient: If True, processes simulations one at a time without 
                             storing entire DataFrames in memory (recommended for large datasets)
            process_all_dates: If True, process all dates in the simulation instead of only
                               the most recent date. Useful for time series analysis.
        
        Returns:
            Dictionary containing simulation results and RWA calculations
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' does not exist")
        
        # Use all calculators if none specified
        if calculator_names is None:
            calculator_names = list(self.calculators.keys())
        
        # Validate calculators
        for name in calculator_names:
            if name not in self.calculators:
                raise ValueError(f"Calculator '{name}' does not exist")
        
        scenario = self.scenarios[scenario_name]
        simulator = scenario['simulator']
        n_iterations = scenario['n_iterations']
        
        # Create a results container for this scenario
        self.results[scenario_name] = {
            'n_iterations': n_iterations,
            'calculator_results': {calc_name: {'results': [], 'calculation_time': 0} 
                                  for calc_name in calculator_names}
        }
        
        self.logger.info(f"[CHECKPOINT] Starting scenario '{scenario_name}' with {n_iterations} iterations")
        start_time = time.time()
        
        if memory_efficient:
            # Memory-efficient approach: process one iteration at a time
            # Instead of storing all simulations, we'll run and process them one by one
            
            for iteration in range(n_iterations):
                # Run a single Monte Carlo iteration in memory-efficient mode
                # We use run_monte_carlo with memory_efficient=True for consistent interface
                # The simulator optimizes this internally
                sim_df = simulator.run_monte_carlo(
                    num_iterations=1, 
                    random_seed=random_seed if random_seed is None else random_seed + iteration,
                    memory_efficient=True
                )[0]
                
                # Process this simulation with each calculator
                for calc_name in calculator_names:
                    calculator = self.calculators[calc_name]
                    calc_start_time = time.time()
                    
                    try:
                        # Process simulation DataFrame using the helper method
                        application_df = self._process_simulation_df(
                            sim_df=sim_df,
                            portfolio_filter=portfolio_filter,
                            process_all_dates=process_all_dates
                        )
                        
                        # Calculate RWA
                        result = calculator.calculate(application_df)
                        self.results[scenario_name]['calculator_results'][calc_name]['results'].append(result)
                        
                    except Exception as e:
                        # Keep error logging as it's important to know when calculations fail
                        self.logger.warning(f"[ERROR] Failed in iteration {iteration+1} with calculator {calc_name}: {str(e)}")
                    
                    # Accumulate calculation time
                    calc_time = time.time() - calc_start_time
                    self.results[scenario_name]['calculator_results'][calc_name]['calculation_time'] += calc_time
                
                # Clean up to free memory - essential for large datasets
                del sim_df
                
                # Explicit garbage collection for large datasets
                if iteration % 5 == 0:
                    import gc
                    gc.collect()
                
                # Provide progress update every 5 iterations or for the last one
                if (iteration + 1) % 5 == 0 or iteration == n_iterations - 1:
                    self.logger.info(f"[PROGRESS] Completed {iteration + 1}/{n_iterations} iterations for scenario '{scenario_name}'")
        
        else:
            # Original approach: store all simulations
            simulations = simulator.run_monte_carlo(num_iterations=n_iterations, random_seed=random_seed)
            
            self.results[scenario_name]['raw_simulations'] = simulations
            
            # Apply each calculator to the simulation results
            for calc_name in calculator_names:
                calculator = self.calculators[calc_name]
                calc_start_time = time.time()
                
                # Process each simulation
                rwa_results = []
                
                for i, sim_df in enumerate(simulations):
                    try:
                        # Process simulation DataFrame using the helper method
                        application_df = self._process_simulation_df(
                            sim_df=sim_df,
                            portfolio_filter=portfolio_filter,
                            process_all_dates=process_all_dates
                        )
                        
                        # Calculate RWA
                        result = calculator.calculate(application_df)
                        rwa_results.append(result)
                        
                    except Exception as e:
                        # Keep error logging as it's important to know when calculations fail
                        self.logger.warning(f"[ERROR] Failed in iteration {i+1} with calculator {calc_name}: {str(e)}")
                
                calc_time = time.time() - calc_start_time
                
                # Store calculator results
                self.results[scenario_name]['calculator_results'][calc_name] = {
                    'results': rwa_results,
                    'calculation_time': calc_time
                }
                
                self.logger.info(f"[CHECKPOINT] Completed calculations with '{calc_name}' in {calc_time:.2f} seconds")
        
        simulation_time = time.time() - start_time
        self.results[scenario_name]['simulation_time'] = simulation_time
        self.logger.info(f"[CHECKPOINT] Completed {n_iterations} simulations in {simulation_time:.2f} seconds")
        
        return self.results[scenario_name]
    
    def get_summary_stats(self, 
                         scenario_name: str, 
                         calculator_name: str) -> Dict[str, float]:
        """
        Get statistical summary of simulation results.
        
        Args:
            scenario_name: Name of the scenario
            calculator_name: Name of the calculator
        
        Returns:
            Dictionary with summary statistics
        """
        self._validate_scenario_calculator(scenario_name, calculator_name)
        
        results = self._get_calculator_results(scenario_name, calculator_name)
        
        if not results:
            return {}
        
        # Extract total RWA from each simulation
        rwa_values = [r.total_rwa for r in results if hasattr(r, 'total_rwa')]
        
        if not rwa_values:
            return {}
        
        return {
            'mean': np.mean(rwa_values),
            'median': np.median(rwa_values),
            'std': np.std(rwa_values),
            'min': np.min(rwa_values),
            'max': np.max(rwa_values),
            'count': len(rwa_values)
        }
    
    def get_percentiles(self, 
                       scenario_name: str, 
                       calculator_name: str, 
                       percentiles: Tuple[float, ...] = (5, 25, 50, 75, 95)) -> Dict[float, float]:
        """
        Get specific percentiles from the result distribution.
        
        Args:
            scenario_name: Name of the scenario
            calculator_name: Name of the calculator
            percentiles: Tuple of percentiles to calculate (values between 0-100)
        
        Returns:
            Dictionary mapping percentiles to their values
        """
        self._validate_scenario_calculator(scenario_name, calculator_name)
        
        results = self._get_calculator_results(scenario_name, calculator_name)
        
        if not results:
            return {}
        
        # Extract total RWA from each simulation
        rwa_values = [r.total_rwa for r in results if hasattr(r, 'total_rwa')]
        
        if not rwa_values:
            return {}
        
        percentile_values = np.percentile(rwa_values, percentiles)
        return dict(zip(percentiles, percentile_values))
    
    def compare_scenarios(self, 
                         scenario_names: List[str], 
                         calculator_name: str) -> pd.DataFrame:
        """
        Calculate differences between scenarios for a given calculator.
        
        Args:
            scenario_names: List of scenario names to compare
            calculator_name: Name of the calculator to use
        
        Returns:
            DataFrame with comparison metrics
        """
        # Validate inputs
        for name in scenario_names:
            self._validate_scenario_calculator(name, calculator_name)
        
        if len(scenario_names) < 2:
            raise ValueError("Need at least two scenarios to compare")
        
        # Get summary stats for each scenario
        stats = []
        for name in scenario_names:
            scenario_stats = self.get_summary_stats(name, calculator_name)
            if scenario_stats:
                scenario_stats['scenario'] = name
                stats.append(scenario_stats)
        
        if not stats:
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(stats)
        
        # Set the first scenario as the baseline
        baseline = comparison_df.iloc[0]
        baseline_name = baseline['scenario']
        
        # Calculate absolute and percentage differences
        for i in range(1, len(comparison_df)):
            scenario = comparison_df.iloc[i]
            scenario_name = scenario['scenario']
            
            comparison_df.loc[i, 'abs_diff_from_baseline'] = scenario['mean'] - baseline['mean']
            comparison_df.loc[i, 'pct_diff_from_baseline'] = \
                (scenario['mean'] / baseline['mean'] - 1) * 100
        
        # Reorder columns for better readability
        column_order = [
            'scenario', 'mean', 'median', 'std', 'min', 'max', 
            'abs_diff_from_baseline', 'pct_diff_from_baseline', 'count'
        ]
        
        # Only include columns that exist
        columns = [col for col in column_order if col in comparison_df.columns]
        
        return comparison_df[columns]
    
    def get_rwa_distribution(self, 
                            scenario_name: str, 
                            calculator_name: str) -> pd.Series:
        """
        Get the full distribution of RWA values across simulations.
        
        Args:
            scenario_name: Name of the scenario
            calculator_name: Name of the calculator
        
        Returns:
            Series containing all RWA values from the simulations
        """
        self._validate_scenario_calculator(scenario_name, calculator_name)
        
        results = self._get_calculator_results(scenario_name, calculator_name)
        
        if not results:
            return pd.Series()
        
        # Extract total RWA from each simulation
        rwa_values = [r.total_rwa for r in results if hasattr(r, 'total_rwa')]
        
        return pd.Series(rwa_values, name='total_rwa')
    
    def _validate_scenario_calculator(self, scenario_name: str, calculator_name: str) -> None:
        """Helper method to validate scenario and calculator names."""
        if scenario_name not in self.results:
            raise ValueError(f"No results for scenario '{scenario_name}'. Run the scenario first.")
        
        if calculator_name not in self.results[scenario_name]['calculator_results']:
            raise ValueError(f"No results for calculator '{calculator_name}' in scenario '{scenario_name}'")
    
    def _get_calculator_results(self, scenario_name: str, calculator_name: str) -> List[Any]:
        """Helper method to extract calculator results."""
        return self.results[scenario_name]['calculator_results'][calculator_name]['results']
        
    def _process_simulation_df(self, 
                              sim_df: pd.DataFrame, 
                              portfolio_filter: Optional[Callable] = None,
                              process_all_dates: bool = False) -> pd.DataFrame:
        """
        Process a simulation DataFrame to prepare it for RWA calculation.
        
        Args:
            sim_df: Simulation DataFrame to process
            portfolio_filter: Optional function to filter/process the simulation results
            process_all_dates: If False, only process the most recent date. If True,
                              process all dates in the DataFrame.
                              
        Returns:
            Processed DataFrame ready for RWA calculation
        """
        # Make a copy to avoid modifying the original
        processed_df = sim_df.copy()
        
        # Filter by date if needed
        if not process_all_dates:
            most_recent_date = processed_df[self.column_mapping['date']].max()
            processed_df = processed_df[processed_df[self.column_mapping['date']] == most_recent_date]
        
        # Apply additional filtering if provided
        if portfolio_filter:
            processed_df = portfolio_filter(processed_df)
        
        # Rename PD column if needed
        pd_column = self.column_mapping['pd']
        target_pd_column = self.column_mapping['target_pd']
        
        if pd_column in processed_df.columns and pd_column != target_pd_column:
            if target_pd_column not in processed_df.columns:
                processed_df.rename(columns={pd_column: target_pd_column}, inplace=True)
        
        return processed_df