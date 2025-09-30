import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List
import statistics

from irbstudio.simulation.portfolio_simulator import PortfolioSimulator


@pytest.fixture
def sample_portfolio_data():
    """Generate a sample portfolio dataset for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Generate 100 loans over 12 quarters, making sure we have data from 2020-2023
    n_loans = 100
    n_quarters = 12
    
    # Generate loan IDs
    loan_ids = [f'LOAN{i:03d}' for i in range(n_loans)]
    
    # Generate dates (quarterly snapshots)
    start_date = datetime(2020, 1, 1)
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
            
            # Random default flag with higher probability to ensure we have some defaults
            is_default = 1 if np.random.random() < 0.05 else 0
            
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


@pytest.fixture
def prepared_simulator(sample_portfolio_data, score_to_rating_bounds):
    """Returns a prepared simulator ready for Monte Carlo runs"""
    simulator = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        target_auc=0.75,
        application_start_date=datetime(2021, 7, 1),
        random_seed=42
    )
    simulator.prepare_simulation()
    return simulator


class TestMonteCarlo:
    """Tests for Monte Carlo simulation functionality."""
    
    def test_monte_carlo_basic_execution(self, prepared_simulator):
        """Test that Monte Carlo simulation runs without errors and returns expected format."""
        num_iterations = 5
        results = prepared_simulator.run_monte_carlo(num_iterations=num_iterations)
        
        # Check correct number of results
        assert len(results) == num_iterations
        
        # Check each result is properly formatted
        for result in results:
            assert isinstance(result, pd.DataFrame)
            assert 'simulated_rating' in result.columns
            assert 'simulated_pd' in result.columns
            assert 'observed_pd' in result.columns
    
    def test_monte_carlo_reproducibility(self, sample_portfolio_data, score_to_rating_bounds):
        """Test that Monte Carlo simulations with same seed produce mostly consistent results."""
        # Create two simulators with the same seed
        simulator1 = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='date',
            default_col='is_default',
            into_default_flag_col='into_default',
            score_col='score',
            target_auc=0.75,
            application_start_date=datetime(2021, 7, 1),
            random_seed=100
        )
        
        simulator2 = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='date',
            default_col='is_default',
            into_default_flag_col='into_default',
            score_col='score',
            target_auc=0.75,
            application_start_date=datetime(2021, 7, 1),
            random_seed=100
        )
        
        # Run Monte Carlo simulations
        results1 = simulator1.prepare_simulation().run_monte_carlo(num_iterations=3, random_seed=50)
        results2 = simulator2.prepare_simulation().run_monte_carlo(num_iterations=3, random_seed=50)
        
        # Results should be very similar with the same seed
        # Use statistical validation rather than exact equality
        for i in range(len(results1)):
            # Check distribution of ratings is similar
            rating_counts1 = results1[i]['simulated_rating'].value_counts().sort_index()
            rating_counts2 = results2[i]['simulated_rating'].value_counts().sort_index()
            
            # Check that the rating counts are close (within 5% difference)
            for rating in rating_counts1.index:
                if rating in rating_counts2.index:
                    count1 = rating_counts1[rating]
                    count2 = rating_counts2[rating]
                    # Allow for some difference due to random factors
                    max_diff_pct = abs(count1 - count2) / max(1, count1) * 100
                    assert max_diff_pct < 5.0, f"Rating {rating} differs by {max_diff_pct:.2f}%"
            
            # Check mean PD is similar
            mean_pd1 = results1[i]['simulated_pd'].mean()
            mean_pd2 = results2[i]['simulated_pd'].mean()
            pd_diff_pct = abs(mean_pd1 - mean_pd2) / mean_pd1 * 100
            assert pd_diff_pct < 5.0, f"Mean PD differs by {pd_diff_pct:.2f}%"
    
    def test_monte_carlo_different_seeds(self, sample_portfolio_data, score_to_rating_bounds):
        """Test that Monte Carlo simulations with different seeds produce different results."""
        simulator1 = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='date',
            default_col='is_default',
            into_default_flag_col='into_default',
            score_col='score',
            target_auc=0.75,
            application_start_date=datetime(2021, 7, 1),
            random_seed=101
        )
        
        simulator2 = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='date',
            default_col='is_default',
            into_default_flag_col='into_default',
            score_col='score',
            target_auc=0.75,
            application_start_date=datetime(2021, 7, 1),
            random_seed=202
        )
        
        # Run Monte Carlo simulations
        results1 = simulator1.prepare_simulation().run_monte_carlo(num_iterations=3)
        results2 = simulator2.prepare_simulation().run_monte_carlo(num_iterations=3)
        
        # Results should be different with different seeds
        for i in range(len(results1)):
            assert not results1[i]['simulated_rating'].equals(results2[i]['simulated_rating'])
    
    def test_monte_carlo_statistical_convergence(self, prepared_simulator):
        """Test that Monte Carlo simulations statistically converge with increasing iterations."""
        # Run two simulations with different numbers of iterations
        results_small = prepared_simulator.run_monte_carlo(num_iterations=5)
        results_large = prepared_simulator.run_monte_carlo(num_iterations=20)
        
        # Calculate the average PD across simulations for each set
        def calculate_avg_pd_by_rating(results: List[pd.DataFrame]) -> dict:
            """Calculate the average PD for each rating across all simulations."""
            all_pds = {}
            
            # Collect PDs by rating across all simulations
            for result in results:
                for rating, rating_group in result.groupby('simulated_rating'):
                    if rating not in all_pds:
                        all_pds[rating] = []
                    all_pds[rating].append(rating_group['simulated_pd'].mean())
            
            # Calculate average PD for each rating
            avg_pds = {rating: sum(pds) / len(pds) for rating, pds in all_pds.items()}
            return avg_pds
        
        # Calculate standard deviation of PDs by rating
        def calculate_pd_std_by_rating(results: List[pd.DataFrame]) -> dict:
            """Calculate the standard deviation of PDs for each rating across simulations."""
            all_pds = {}
            
            # Collect PDs by rating across all simulations
            for result in results:
                for rating, rating_group in result.groupby('simulated_rating'):
                    if rating not in all_pds:
                        all_pds[rating] = []
                    all_pds[rating].append(rating_group['simulated_pd'].mean())
            
            # Calculate standard deviation for each rating
            std_pds = {rating: statistics.stdev(pds) if len(pds) > 1 else 0
                      for rating, pds in all_pds.items()}
            return std_pds
        
        # Calculate average PDs and standard deviations
        avg_pds_small = calculate_avg_pd_by_rating(results_small)
        avg_pds_large = calculate_avg_pd_by_rating(results_large)
        std_pds_small = calculate_pd_std_by_rating(results_small)
        std_pds_large = calculate_pd_std_by_rating(results_large)
        
        # Check that the standard deviation decreases with more iterations
        # (not for all ratings as some may not have enough samples, but in general)
        common_ratings = set(std_pds_small.keys()) & set(std_pds_large.keys())
        common_ratings = [r for r in common_ratings if r != 'D']  # Exclude default rating as it's always 1.0
        
        if common_ratings:  # Only check if we have common ratings to compare
            avg_std_small = sum(std_pds_small[r] for r in common_ratings) / len(common_ratings)
            avg_std_large = sum(std_pds_large[r] for r in common_ratings) / len(common_ratings)
            
            # For statistical convergence, standard deviation should decrease with more samples
            # Allow for small fluctuations or equal values due to randomness
            assert avg_std_large <= avg_std_small * 1.2  # 20% tolerance
    
    def test_monte_carlo_performance(self, prepared_simulator):
        """Test the performance of Monte Carlo simulations."""
        # Measure time to run 10 iterations
        start_time = time.time()
        prepared_simulator.run_monte_carlo(num_iterations=10)
        duration = time.time() - start_time
        
        # Average time per iteration should be reasonable (adjust this threshold based on your system)
        avg_time_per_iteration = duration / 10
        print(f"Average time per iteration: {avg_time_per_iteration:.3f} seconds")
        
        # Ensure each iteration takes less than 5 seconds on average (adjust as needed)
        assert avg_time_per_iteration < 5.0, f"Performance too slow: {avg_time_per_iteration:.2f} seconds per iteration"
    
    def test_monte_carlo_rating_distribution(self, prepared_simulator):
        """Test that the rating distribution is stable across multiple simulations."""
        # Run multiple simulations
        results = prepared_simulator.run_monte_carlo(num_iterations=10)
        
        # Calculate rating distribution for each simulation
        rating_distributions = []
        for result in results:
            dist = result['simulated_rating'].value_counts(normalize=True)
            rating_distributions.append(dist)
        
        # Calculate the average distribution
        all_ratings = set()
        for dist in rating_distributions:
            all_ratings.update(dist.index)
        
        # Create a DataFrame with all rating distributions
        dist_df = pd.DataFrame(rating_distributions).fillna(0)
        
        # Calculate the standard deviation for each rating
        std_by_rating = dist_df.std()
        
        # Check that the standard deviation is relatively small for all ratings
        # This indicates stability in the rating distribution across simulations
        for rating, std in std_by_rating.items():
            assert std < 0.1, f"Rating {rating} has unstable distribution across simulations: std={std:.4f}"
    
    def test_monte_carlo_aggregation(self, prepared_simulator):
        """Test that we can aggregate results from multiple Monte Carlo simulations."""
        # Run multiple simulations
        num_iterations = 5
        results = prepared_simulator.run_monte_carlo(num_iterations=num_iterations)
        
        # Aggregate results by calculating average metrics across all simulations
        def aggregate_simulations(results: List[pd.DataFrame]) -> dict:
            """Calculate aggregate statistics from multiple Monte Carlo simulations."""
            # Extract the latest date (application date) for each simulation
            latest_date = results[0]['date'].max()
            
            # Filter to only include the application date records
            application_results = [result[result['date'] == latest_date] for result in results]
            
            # Calculate average metrics
            avg_metrics = {}
            
            # Average PD by rating
            all_pds_by_rating = {}
            for result in application_results:
                for rating, group in result.groupby('simulated_rating'):
                    if rating not in all_pds_by_rating:
                        all_pds_by_rating[rating] = []
                    all_pds_by_rating[rating].append(group['simulated_pd'].mean())
            
            avg_metrics['avg_pd_by_rating'] = {
                rating: sum(pds) / len(pds) for rating, pds in all_pds_by_rating.items()
            }
            
            # Average exposure by rating
            all_exp_by_rating = {}
            for result in application_results:
                for rating, group in result.groupby('simulated_rating'):
                    if rating not in all_exp_by_rating:
                        all_exp_by_rating[rating] = []
                    all_exp_by_rating[rating].append(group['exposure'].sum())
            
            avg_metrics['avg_exposure_by_rating'] = {
                rating: sum(exps) / len(exps) for rating, exps in all_exp_by_rating.items()
            }
            
            # Overall average PD
            avg_metrics['overall_avg_pd'] = sum(
                result['simulated_pd'].mean() for result in application_results
            ) / len(application_results)
            
            return avg_metrics
        
        # Aggregate the simulation results
        aggregated_results = aggregate_simulations(results)
        
        # Check that the aggregation contains the expected metrics
        assert 'avg_pd_by_rating' in aggregated_results
        assert 'avg_exposure_by_rating' in aggregated_results
        assert 'overall_avg_pd' in aggregated_results
        
        # Check that the aggregated metrics are reasonable
        assert 0 < aggregated_results['overall_avg_pd'] < 1
        for rating, avg_pd in aggregated_results['avg_pd_by_rating'].items():
            if rating == 'D':
                assert avg_pd == 1.0
            else:
                assert 0 <= avg_pd <= 1.0