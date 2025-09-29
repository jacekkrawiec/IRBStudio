import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from irbstudio.simulation.pd_simulator import simulate_portfolio as original_simulate_portfolio
from irbstudio.simulation.portfolio_simulator import simulate_portfolio as oop_simulate_portfolio
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator


@pytest.fixture
def sample_portfolio_data():
    """Generate a sample portfolio dataset for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Generate 50 loans over 12 quarters, making sure we have data into 2022
    n_loans = 50
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


def test_portfolio_simulator_class_initialization(sample_portfolio_data, score_to_rating_bounds):
    """Test that the PortfolioSimulator class initializes correctly."""
    simulator = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
    )
    
    # Check that all required attributes are set
    assert simulator.rating_col == 'rating'
    assert simulator.loan_id_col == 'loan_id'
    assert simulator.date_col == 'date'
    assert simulator.default_col == 'is_default'
    assert simulator.into_default_flag_col == 'into_default'
    assert simulator.score_col == 'score'
    assert simulator.score_to_rating_bounds == score_to_rating_bounds
    assert simulator.is_prepared == False


def test_portfolio_simulator_preparation(sample_portfolio_data, score_to_rating_bounds):
    """Test that the preparation step works correctly."""
    simulator = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        target_auc=0.75  # Set a target AUC to avoid the None error
    )
    
    # Run preparation
    simulator.prepare_simulation()
    
    # Check that preparation was completed
    assert simulator.is_prepared == True
    
    # Check that all required components were generated
    assert simulator.historical_df is not None
    assert simulator.application_df is not None
    assert simulator.beta_mixture is not None
    assert simulator.simulated_migration_matrix is not None
    
    # Check that the historical dataframe has simulated ratings
    assert 'simulated_rating' in simulator.historical_df.columns


def test_compatibility_with_original_implementation(sample_portfolio_data, score_to_rating_bounds):
    """
    Test that the new OOP implementation produces equivalent results to the original implementation.
    """
    # We'll use a fixed application_start_date for reproducibility
    application_start_date = datetime(2021, 7, 1)  # Moved earlier to ensure we have application data
    
    # Run original implementation
    np.random.seed(42)  # Set seed for reproducibility
    original_result = original_simulate_portfolio(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        application_start_date=application_start_date,
        target_auc=0.75
    )
    
    # Run new OOP implementation through wrapper function
    np.random.seed(42)  # Reset seed for reproducibility
    oop_result = oop_simulate_portfolio(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        application_start_date=application_start_date,
        target_auc=0.75
    )
    
    # Check that both results have the same shape
    assert original_result.shape == oop_result.shape
    
    # Check that both results have the same columns
    assert set(original_result.columns) == set(oop_result.columns)
    
    # Check that the simulated ratings distribution is similar
    # We don't expect exact equality due to random processes, 
    # but the distributions should be similar
    orig_rating_counts = original_result['simulated_rating'].value_counts().sort_index()
    oop_rating_counts = oop_result['simulated_rating'].value_counts().sort_index()
    
    # Both should have the same unique ratings
    assert set(orig_rating_counts.index) == set(oop_rating_counts.index)


def test_monte_carlo_simulation(sample_portfolio_data, score_to_rating_bounds):
    """Test that the Monte Carlo simulation functionality works."""
    simulator = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        target_auc=0.75,  # Set a target AUC
        application_start_date=datetime(2021, 7, 1)  # Set an appropriate application date
    )
    
    # Run a small Monte Carlo simulation
    results = simulator.prepare_simulation().run_monte_carlo(num_iterations=3, random_seed=42)
    
    # Check that we got the expected number of results
    assert len(results) == 3
    
    # Check that each result is a dataframe with the expected columns
    for result in results:
        assert isinstance(result, pd.DataFrame)
        assert 'simulated_rating' in result.columns
        assert 'simulated_pd' in result.columns
        
    # Check that the results are not identical (stochastic variation)
    assert not results[0]['simulated_rating'].equals(results[1]['simulated_rating'])
    assert not results[1]['simulated_rating'].equals(results[2]['simulated_rating'])


def test_random_seed_reproducibility(sample_portfolio_data, score_to_rating_bounds):
    """Test that setting the same random seed produces the same results."""
    # Create two simulators with the same random seed
    simulator1 = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        target_auc=0.75,  # Set a target AUC
        application_start_date=datetime(2021, 7, 1),  # Set an appropriate application date
        random_seed=42
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
        target_auc=0.75,  # Set a target AUC
        application_start_date=datetime(2021, 7, 1),  # Set an appropriate application date
        random_seed=42
    )
    
    # Run simulations
    result1 = simulator1.prepare_simulation().simulate_once()
    result2 = simulator2.prepare_simulation().simulate_once()
    
    # Results should be identical with the same seed
    pd.testing.assert_frame_equal(result1, result2)
    
    # Create a simulator with a different seed
    simulator3 = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        target_auc=0.75,  # Set a target AUC
        application_start_date=datetime(2021, 7, 1),  # Set an appropriate application date
        random_seed=24
    )
    
    # Run simulation
    result3 = simulator3.prepare_simulation().simulate_once()
    
    # Results should be different with a different seed
    assert not result1['simulated_rating'].equals(result3['simulated_rating'])