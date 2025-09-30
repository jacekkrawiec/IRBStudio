import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple

from irbstudio.simulation.portfolio_simulator import PortfolioSimulator


@pytest.fixture
def large_portfolio_data():
    """Generate a larger sample portfolio dataset for performance benchmarking."""
    np.random.seed(42)  # For reproducible tests
    
    # Generate 500 loans over 20 quarters for a substantial dataset
    n_loans = 500
    n_quarters = 20
    
    # Generate loan IDs
    loan_ids = [f'LOAN{i:03d}' for i in range(n_loans)]
    
    # Generate dates (quarterly snapshots)
    start_date = datetime(2018, 1, 1)
    dates = [start_date + timedelta(days=90*i) for i in range(n_quarters)]
    
    # Create all combinations of loan_ids and dates
    data = []
    for loan_id in loan_ids:
        for date in dates:
            # Scores between 0 and 1 with beta distribution
            score = np.random.beta(2, 5)
            
            # Generate a rating based on score
            if score < 0.05:
                rating = '1'
            elif score < 0.10:
                rating = '2'
            elif score < 0.20:
                rating = '3'
            elif score < 0.30:
                rating = '4'
            else:
                rating = '5'
            
            # Random default flag with low probability for realism
            is_default = 1 if np.random.random() < 0.03 else 0
            
            # If defaulted, set rating to 'D' and score to 1.0
            if is_default == 1:
                rating = 'D'
                score = 1.0
            
            # Generate a row
            data.append({
                'loan_id': loan_id,
                'date': date,
                'score': score,
                'rating': rating,
                'is_default': is_default,
                'into_default': 1 if is_default == 1 and date > start_date else 0,
                'exposure': np.random.uniform(50000, 500000),
                'ltv': np.random.uniform(0.4, 0.95)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def score_to_rating_bounds():
    """Define rating bounds for testing."""
    return {
        '1': (0, 0.05),
        '2': (0.05, 0.10),
        '3': (0.10, 0.20),
        '4': (0.20, 0.30),
        '5': (0.30, 1.0),
        'D': (0, 0)  # Default rating (handled separately)
    }


class TestMonteCarloPerformance:
    """Benchmarking tests for Monte Carlo performance."""
    
    def test_iteration_scaling(self, large_portfolio_data, score_to_rating_bounds):
        """Test how simulation time scales with number of iterations."""
        # Skip this test in regular test runs to save time
        if not os.environ.get('RUN_BENCHMARKS', False):
            pytest.skip("Skipping benchmark test. Set RUN_BENCHMARKS=1 to run.")
            
        application_start_date = datetime(2022, 7, 1)
        
        # Initialize the simulator
        simulator = PortfolioSimulator(
            portfolio_df=large_portfolio_data,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='date',
            default_col='is_default',
            into_default_flag_col='into_default',
            score_col='score',
            application_start_date=application_start_date,
            target_auc=0.75,
            random_seed=42
        )
        
        # Prepare the simulator (do this once outside the timing loop)
        print("\nPreparing simulator...")
        prep_start_time = time.time()
        simulator.prepare_simulation()
        prep_time = time.time() - prep_start_time
        print(f"Preparation time: {prep_time:.2f} seconds")
        
        # Test different numbers of iterations
        iterations_to_test = [1, 2, 5, 10, 20]
        times = []
        
        for num_iterations in iterations_to_test:
            print(f"Testing {num_iterations} iterations...")
            start_time = time.time()
            simulator.run_monte_carlo(num_iterations=num_iterations)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"Completed in {elapsed:.2f} seconds ({elapsed/num_iterations:.2f} seconds per iteration)")
        
        # Create a plot of iterations vs. time
        plt.figure(figsize=(10, 6))
        plt.plot(iterations_to_test, times, marker='o', linestyle='-')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Time (seconds)')
        plt.title('Monte Carlo Performance Scaling')
        plt.grid(True)
        
        # Add a trend line
        z = np.polyfit(iterations_to_test, times, 1)
        p = np.poly1d(z)
        plt.plot(iterations_to_test, p(iterations_to_test), "r--", 
                 label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}")
        plt.legend()
        
        # Save the plot
        os.makedirs('benchmark_results', exist_ok=True)
        plt.savefig('benchmark_results/monte_carlo_scaling.png')
        
        # Check that the time scales roughly linearly with iterations
        # This confirms our Monte Carlo implementation has good performance characteristics
        correlation = np.corrcoef(iterations_to_test, times)[0, 1]
        print(f"Correlation coefficient: {correlation:.4f}")
        assert correlation > 0.95, "Time should scale linearly with number of iterations"
    
    def test_data_size_scaling(self, large_portfolio_data, score_to_rating_bounds):
        """Test how simulation performance scales with data size."""
        # Skip this test in regular test runs to save time
        if not os.environ.get('RUN_BENCHMARKS', False):
            pytest.skip("Skipping benchmark test. Set RUN_BENCHMARKS=1 to run.")
            
        application_start_date = datetime(2022, 7, 1)
        
        # Test different dataset sizes (percentage of full dataset)
        sizes_to_test = [0.1, 0.2, 0.5, 0.8, 1.0]
        prep_times = []
        sim_times = []
        
        for size_factor in sizes_to_test:
            # Create a subset of the data
            if size_factor < 1.0:
                sample_size = int(len(large_portfolio_data) * size_factor)
                subset_data = large_portfolio_data.sample(sample_size, random_state=42)
            else:
                subset_data = large_portfolio_data
            
            print(f"\nTesting with {len(subset_data)} rows ({size_factor*100:.0f}% of full dataset)")
            
            # Initialize the simulator
            simulator = PortfolioSimulator(
                portfolio_df=subset_data,
                score_to_rating_bounds=score_to_rating_bounds,
                rating_col='rating',
                loan_id_col='loan_id',
                date_col='date',
                default_col='is_default',
                into_default_flag_col='into_default',
                score_col='score',
                application_start_date=application_start_date,
                target_auc=0.75,
                random_seed=42
            )
            
            # Measure preparation time
            prep_start_time = time.time()
            simulator.prepare_simulation()
            prep_elapsed = time.time() - prep_start_time
            prep_times.append(prep_elapsed)
            print(f"Preparation time: {prep_elapsed:.2f} seconds")
            
            # Measure simulation time (fixed 3 iterations)
            sim_start_time = time.time()
            simulator.run_monte_carlo(num_iterations=3)
            sim_elapsed = time.time() - sim_start_time
            sim_times.append(sim_elapsed)
            print(f"Simulation time (3 iterations): {sim_elapsed:.2f} seconds")
        
        # Create plots for preparation and simulation scaling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot preparation time scaling
        data_sizes = [len(large_portfolio_data) * size for size in sizes_to_test]
        ax1.plot(data_sizes, prep_times, marker='o', linestyle='-')
        ax1.set_xlabel('Dataset Size (rows)')
        ax1.set_ylabel('Preparation Time (seconds)')
        ax1.set_title('Preparation Time vs. Dataset Size')
        ax1.grid(True)
        
        # Plot simulation time scaling
        ax2.plot(data_sizes, sim_times, marker='o', linestyle='-')
        ax2.set_xlabel('Dataset Size (rows)')
        ax2.set_ylabel('Simulation Time (seconds)')
        ax2.set_title('Simulation Time vs. Dataset Size')
        ax2.grid(True)
        
        # Save the plots
        os.makedirs('benchmark_results', exist_ok=True)
        plt.tight_layout()
        plt.savefig('benchmark_results/data_scaling.png')
        
        # Basic performance assertions
        max_prep_time_per_row = prep_times[-1] / data_sizes[-1] * 1000  # ms per row
        max_sim_time_per_row = sim_times[-1] / data_sizes[-1] * 1000    # ms per row
        
        print(f"\nPerformance metrics:")
        print(f"Preparation: {max_prep_time_per_row:.3f} ms per row")
        print(f"Simulation: {max_sim_time_per_row:.3f} ms per row")
        
        # These thresholds would need to be adjusted based on the expected performance
        assert max_prep_time_per_row < 0.5, f"Preparation too slow: {max_prep_time_per_row:.3f} ms per row"
        assert max_sim_time_per_row < 0.5, f"Simulation too slow: {max_sim_time_per_row:.3f} ms per row"
    
    def test_memory_usage(self, large_portfolio_data, score_to_rating_bounds):
        """Test the memory efficiency of Monte Carlo simulations."""
        # Skip this test in regular test runs to save time
        if not os.environ.get('RUN_BENCHMARKS', False):
            pytest.skip("Skipping benchmark test. Set RUN_BENCHMARKS=1 to run.")
            
        # This test is more qualitative and relies on process monitoring
        # We'll just run a simulation and track the memory usage before and after
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Force garbage collection
        gc.collect()
        
        # Get baseline memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"\nInitial memory usage: {initial_memory:.2f} MB")
        
        # Initialize and prepare the simulator
        application_start_date = datetime(2022, 7, 1)
        simulator = PortfolioSimulator(
            portfolio_df=large_portfolio_data,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='date',
            default_col='is_default',
            into_default_flag_col='into_default',
            score_col='score',
            application_start_date=application_start_date,
            target_auc=0.75,
            random_seed=42
        )
        
        # Measure memory after initialization
        simulator.prepare_simulation()
        after_prep_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after preparation: {after_prep_memory:.2f} MB (Δ: {after_prep_memory - initial_memory:.2f} MB)")
        
        # Run a single simulation
        _ = simulator.simulate_once()
        after_single_sim_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after single simulation: {after_single_sim_memory:.2f} MB (Δ: {after_single_sim_memory - after_prep_memory:.2f} MB)")
        
        # Run multiple simulations
        _ = simulator.run_monte_carlo(num_iterations=10)
        after_multi_sim_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory after 10 simulations: {after_multi_sim_memory:.2f} MB (Δ: {after_multi_sim_memory - after_single_sim_memory:.2f} MB)")
        
        # Check for memory leaks (significant increase after running multiple simulations)
        memory_growth_per_iteration = (after_multi_sim_memory - after_single_sim_memory) / 10
        print(f"Memory growth per iteration: {memory_growth_per_iteration:.2f} MB")
        
        # A properly implemented simulation should have minimal memory growth per iteration
        # This is a qualitative check - the actual threshold depends on dataset size
        assert memory_growth_per_iteration < 20, f"Possible memory leak: {memory_growth_per_iteration:.2f} MB growth per iteration"


def run_benchmark_suite(large_portfolio_data, score_to_rating_bounds):
    """Function to run all benchmarks for profiling."""
    # This function is meant to be called from a script for profiling purposes
    application_start_date = datetime(2022, 7, 1)
    
    # Initialize the simulator
    simulator = PortfolioSimulator(
        portfolio_df=large_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        application_start_date=application_start_date,
        target_auc=0.75,
        random_seed=42
    )
    
    # Prepare the simulator
    simulator.prepare_simulation()
    
    # Run simulations with different numbers of iterations
    simulator.run_monte_carlo(num_iterations=1)
    simulator.run_monte_carlo(num_iterations=5)
    simulator.run_monte_carlo(num_iterations=10)
    
    return "Benchmarking complete"