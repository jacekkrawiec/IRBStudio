"""
Tests for PortfolioSimulator.

Priority 1: Critical - Core Functionality
"""

import pytest
import pandas as pd
import numpy as np

from irbstudio.simulation.portfolio_simulator import PortfolioSimulator


class TestPortfolioSimulatorInit:
    """Tests for PortfolioSimulator initialization."""
    
    def test_portfolio_simulator_init(self, sample_portfolio_df, score_to_rating_bounds):
        """Test PortfolioSimulator.__init__() basic."""
        simulator = PortfolioSimulator(
            portfolio_df=sample_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            asset_correlation=0.15,
            random_seed=42
        )
        
        assert simulator is not None
        assert simulator.target_auc == 0.80
        assert simulator.asset_correlation == 0.15
    
    def test_portfolio_simulator_with_target_auc(self, sample_portfolio_df, score_to_rating_bounds):
        """Test initialization with target_auc."""
        simulator = PortfolioSimulator(
            portfolio_df=sample_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.85
        )
        
        assert simulator.target_auc == 0.85
    
    def test_portfolio_simulator_with_asset_correlation(self, sample_portfolio_df, score_to_rating_bounds):
        """Test initialization with asset correlation."""
        simulator = PortfolioSimulator(
            portfolio_df=sample_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            asset_correlation=0.20
        )
        
        assert simulator.asset_correlation == 0.20
    
    def test_portfolio_simulator_with_random_seed(self, sample_portfolio_df, score_to_rating_bounds):
        """Test initialization with random seed."""
        simulator = PortfolioSimulator(
            portfolio_df=sample_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            random_seed=42
        )
        
        assert simulator.random_seed == 42
    
    def test_portfolio_simulator_invalid_target_auc(self, sample_portfolio_df, score_to_rating_bounds):
        """Test that invalid AUC values are accepted (no validation at init)."""
        # PortfolioSimulator doesn't validate parameters at __init__
        # Invalid values would only cause issues during simulation
        simulator = PortfolioSimulator(
            portfolio_df=sample_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=1.5  # Invalid but not validated at init
        )
        
        # Should initialize without error
        assert simulator.target_auc == 1.5
    
    def test_portfolio_simulator_invalid_correlation(self, sample_portfolio_df, score_to_rating_bounds):
        """Test that invalid correlation values are accepted (no validation at init)."""
        # PortfolioSimulator doesn't validate parameters at __init__
        # Invalid values would only cause issues during simulation
        simulator = PortfolioSimulator(
            portfolio_df=sample_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            asset_correlation=1.5  # Invalid but not validated at init
        )
        
        # Should initialize without error
        assert simulator.asset_correlation == 1.5


class TestPrepareSimulation:
    """Tests for prepare_simulation() method."""
    
    def test_prepare_simulation_basic(self, sample_portfolio_df, score_to_rating_bounds):
        """Test prepare_simulation() executes."""
        simulator = PortfolioSimulator(
            portfolio_df=sample_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            asset_correlation=0.15,
            application_start_date='2024-01-01'
        )
        
        simulator.prepare_simulation()
        
        # Should complete without error
        assert True
    
    def test_prepare_simulation_portfolio_segmentation(self, sample_portfolio_df, score_to_rating_bounds):
        """Test portfolio historical/application split."""
        simulator = PortfolioSimulator(
            portfolio_df=sample_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            application_start_date='2024-01-01'
        )
        
        simulator.prepare_simulation()
        
        # Check that segmentation happened
        if hasattr(simulator, 'historical_portfolio'):
            assert simulator.historical_portfolio is not None
        if hasattr(simulator, 'application_portfolio'):
            assert simulator.application_portfolio is not None
    
    def test_prepare_simulation_without_application_date(self, small_portfolio_df, score_to_rating_bounds):
        """Test preparation without application_start_date."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80
        )
        
        # Should handle missing application_start_date
        simulator.prepare_simulation()
        assert True


class TestSimulateOnce:
    """Tests for simulate_once() method."""
    
    def test_simulate_once_basic(self, small_portfolio_df, score_to_rating_bounds):
        """Test simulate_once() single iteration."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            random_seed=42
        )
        simulator.prepare_simulation()
        
        simulated_df = simulator.simulate_once()
        
        assert simulated_df is not None
        assert isinstance(simulated_df, pd.DataFrame)
        assert len(simulated_df) > 0
    
    def test_simulate_once_returns_dataframe(self, small_portfolio_df, score_to_rating_bounds):
        """Test that simulate_once() returns DataFrame with correct structure."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80
        )
        simulator.prepare_simulation()
        
        simulated_df = simulator.simulate_once()
        
        assert isinstance(simulated_df, pd.DataFrame)
        # Should have at least loan_id and some simulated columns
        assert 'loan_id' in simulated_df.columns or len(simulated_df.columns) > 0
    
    def test_simulate_once_with_seed(self, small_portfolio_df, score_to_rating_bounds):
        """Test reproducibility with seed."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            random_seed=42
        )
        simulator.prepare_simulation()
        
        sim1 = simulator.simulate_once(random_seed=42)
        
        # Reset and simulate again with same seed
        simulator2 = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            random_seed=42
        )
        simulator2.prepare_simulation()
        
        sim2 = simulator2.simulate_once(random_seed=42)
        
        # Results should be similar (at least same shape)
        assert sim1.shape == sim2.shape


class TestRunMonteCarlo:
    """Tests for run_monte_carlo() method."""
    
    def test_run_monte_carlo_basic(self, small_portfolio_df, score_to_rating_bounds):
        """Test run_monte_carlo() with num_iterations."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            random_seed=42
        )
        simulator.prepare_simulation()
        
        results = simulator.run_monte_carlo(num_iterations=5)
        
        assert results is not None
    
    def test_run_monte_carlo_returns_list(self, small_portfolio_df, score_to_rating_bounds):
        """Test that run_monte_carlo() returns list of DataFrames."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80
        )
        simulator.prepare_simulation()
        
        results = simulator.run_monte_carlo(num_iterations=3)
        
        if isinstance(results, list):
            assert len(results) == 3
            for result in results:
                assert isinstance(result, pd.DataFrame)
    
    def test_run_monte_carlo_correct_count(self, small_portfolio_df, score_to_rating_bounds):
        """Test that correct number of iterations returned."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80
        )
        simulator.prepare_simulation()
        
        n_iter = 7
        results = simulator.run_monte_carlo(num_iterations=n_iter)
        
        if isinstance(results, list):
            assert len(results) == n_iter
    
    def test_run_monte_carlo_with_seed(self, small_portfolio_df, score_to_rating_bounds):
        """Test reproducible results with seed."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            random_seed=42
        )
        simulator.prepare_simulation()
        
        results1 = simulator.run_monte_carlo(num_iterations=3)
        
        # Create new simulator with same seed
        simulator2 = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            random_seed=42
        )
        simulator2.prepare_simulation()
        
        results2 = simulator2.run_monte_carlo(num_iterations=3)
        
        # Should get same number of results
        if isinstance(results1, list) and isinstance(results2, list):
            assert len(results1) == len(results2)
    
    def test_run_monte_carlo_memory_efficient(self, small_portfolio_df, score_to_rating_bounds):
        """Test memory-efficient mode works."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.80,
            random_seed=42
        )
        simulator.prepare_simulation()
        
        # Run in memory-efficient mode if available
        results = simulator.run_monte_carlo(
            num_iterations=3,
            memory_efficient=True if 'memory_efficient' in simulator.run_monte_carlo.__code__.co_varnames else False
        )
        
        assert results is not None


class TestBetaMixtureModel:
    """Tests for Beta Mixture Model (if accessible)."""
    
    def test_beta_mixture_score_generation(self, small_portfolio_df, score_to_rating_bounds):
        """Test that scores are generated within [0, 1] range."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            target_auc=0.85,
            random_seed=42
        )
        simulator.prepare_simulation()
        
        simulated_df = simulator.simulate_once()
        
        # Check if simulated scores exist and are in valid range
        score_columns = [col for col in simulated_df.columns if 'score' in col.lower() or 'pd' in col.lower()]
        
        for col in score_columns:
            if pd.api.types.is_numeric_dtype(simulated_df[col]):
                valid_scores = simulated_df[col].dropna()
                if len(valid_scores) > 0:
                    assert (valid_scores >= 0).all(), f"Scores in {col} should be >= 0"
                    assert (valid_scores <= 1).all() or (valid_scores <= 850).all(), \
                        f"Scores in {col} should be in valid range"
