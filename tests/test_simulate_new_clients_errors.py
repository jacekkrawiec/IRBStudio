"""
Specific tests for the index error bug in _simulate_new_clients.

These tests are designed to catch the intermittent index error:
"arrays used as indices must be of integer or boolean type"

This error occurs when:
1. Rating alignment fails between new_clients_df and migration_matrix
2. NaN/None values appear in rating indices
3. Float dtype is used instead of integer for array indexing
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator


@pytest.fixture
def score_to_rating_bounds():
    """Standard rating bounds."""
    return {
        '1': (-1, 0.003613294451497495),
        '2': (0.003613294451497495, 0.005780360195785761),
        '3': (0.005780360195785761, 0.03225071728229523),
        '4': (0.03225071728229523, 0.039578670635819435),
        '5': (0.039578670635819435, 0.256146103143692),
        '6': (0.256146103143692, 0.7653337121009827),
        '7': (0.7653337121009827, 50)
    }


class TestIndexErrorReproduction:
    """Tests specifically designed to reproduce and prevent the index error."""
    
    def test_multiple_iterations_different_seeds(self, score_to_rating_bounds):
        """
        Run multiple iterations with different seeds to catch index errors.
        This mimics the integrated_analysis.run_scenario behavior.
        """
        # Create portfolio data
        np.random.seed(42)
        dates_hist = pd.date_range('2023-01-31', '2023-12-31', freq='M')
        dates_app = pd.date_range('2024-01-31', '2024-06-30', freq='M')
        
        data = []
        
        # Historical data
        for fid in range(200):
            for date in dates_hist:
                data.append({
                    'loan_id': f'HIST_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3', '4', '5', '6', '7']),
                    'default_flag': 0,
                    'into_default_flag': 0 if np.random.random() > 0.02 else 1,
                })
        
        # New clients
        for fid in range(100):
            for date in dates_app:
                data.append({
                    'loan_id': f'NEW_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3', '4', '5', '6', '7']),
                    'default_flag': 0,
                    'into_default_flag': 0,
                })
        
        portfolio_df = pd.DataFrame(data)
        
        # Run multiple iterations with different seeds (like integrated_analysis does)
        errors = []
        for iteration in range(10):
            seed = 1000 + iteration
            
            try:
                sim = PortfolioSimulator(
                    portfolio_df=portfolio_df,
                    score_to_rating_bounds=score_to_rating_bounds,
                    rating_col='rating',
                    loan_id_col='loan_id',
                    date_col='observation_date',
                    default_col='default_flag',
                    into_default_flag_col='into_default_flag',
                    score_col='score',
                    application_start_date=datetime(2024, 1, 1),
                    asset_correlation=0.15,
                    random_seed=seed
                )
                sim.prepare_simulation()
                
                # Full simulation cycle
                result = sim.simulate_once(random_seed=seed)
                
                # Verify no errors in result
                assert 'simulated_rating' in result.columns
                assert result['simulated_rating'].notna().all()
                
            except (TypeError, IndexError, KeyError) as e:
                errors.append({
                    'iteration': iteration,
                    'seed': seed,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        # Report any errors
        if errors:
            error_msg = f"Index errors occurred in {len(errors)}/10 iterations:\n"
            for err in errors:
                error_msg += f"  Iteration {err['iteration']} (seed={err['seed']}): {err['error_type']}: {err['error']}\n"
            pytest.fail(error_msg)
    
    def test_rating_alignment_check(self, score_to_rating_bounds):
        """Test that all new client ratings exist in migration matrix."""
        np.random.seed(42)
        dates_hist = pd.date_range('2023-01-31', '2023-12-31', freq='M')
        dates_app = pd.date_range('2024-01-31', '2024-03-31', freq='M')
        
        data = []
        for fid in range(100):
            for date in dates_hist:
                data.append({
                    'loan_id': f'HIST_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3', '4', '5']),
                    'default_flag': 0,
                    'into_default_flag': 0 if np.random.random() > 0.02 else 1,
                })
        
        for fid in range(50):
            for date in dates_app:
                data.append({
                    'loan_id': f'NEW_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3', '4', '5']),
                    'default_flag': 0,
                    'into_default_flag': 0,
                })
        
        portfolio_df = pd.DataFrame(data)
        
        sim = PortfolioSimulator(
            portfolio_df=portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='observation_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date=datetime(2024, 1, 1),
            asset_correlation=0.15,
            random_seed=42
        )
        sim.prepare_simulation()
        sim._simulate_historical_ratings()
        sim._calculate_migration_matrix()
        sim._calculate_long_term_pd(use_simulated=True)
        
        result = sim._simulate_new_clients()
        
        # Check alignment
        migration_ratings = set(sim.simulated_migration_matrix.index)
        new_ratings = set(result['simulated_rating'].dropna().unique())
        
        missing_ratings = new_ratings - migration_ratings
        assert len(missing_ratings) == 0, (
            f"Ratings {missing_ratings} in new_clients but not in migration_matrix. "
            f"Migration matrix ratings: {sorted(migration_ratings)}, "
            f"New client ratings: {sorted(new_ratings)}"
        )
    
    def test_no_nan_in_rating_indices(self, score_to_rating_bounds):
        """Test that rating-to-index mapping produces no NaN values."""
        np.random.seed(42)
        dates_hist = pd.date_range('2023-01-31', '2023-12-31', freq='M')
        dates_app = pd.date_range('2024-01-31', '2024-03-31', freq='M')
        
        data = []
        for fid in range(100):
            for date in dates_hist:
                data.append({
                    'loan_id': f'HIST_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3', '4', '5']),
                    'default_flag': 0,
                    'into_default_flag': 0,
                })
        
        for fid in range(50):
            for date in dates_app:
                data.append({
                    'loan_id': f'NEW_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3', '4', '5']),
                    'default_flag': 0,
                    'into_default_flag': 0,
                })
        
        portfolio_df = pd.DataFrame(data)
        
        sim = PortfolioSimulator(
            portfolio_df=portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='observation_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date=datetime(2024, 1, 1),
            asset_correlation=0.15,
            random_seed=42
        )
        sim.prepare_simulation()
        sim._simulate_historical_ratings()
        sim._calculate_migration_matrix()
        sim._calculate_long_term_pd(use_simulated=True)
        
        result = sim._simulate_new_clients()
        
        # Verify no NaN ratings
        assert result['simulated_rating'].notna().all(), "Found NaN in simulated_rating"
    
    def test_extreme_score_values(self, score_to_rating_bounds):
        """Test handling of extreme score values (0, 1, negative, >1)."""
        np.random.seed(42)
        dates_hist = pd.date_range('2023-01-31', '2023-12-31', freq='M')
        dates_app = pd.date_range('2024-01-31', '2024-03-31', freq='M')
        
        data = []
        for fid in range(100):
            for date in dates_hist:
                data.append({
                    'loan_id': f'HIST_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3']),
                    'default_flag': 0,
                    'into_default_flag': 0,
                })
        
        # New clients with extreme scores
        extreme_scores = [0.0, 0.000001, 0.999999, 1.0, -0.001, 1.001]
        for fid, score in enumerate(extreme_scores * 10):  # Repeat to have enough data
            for date in dates_app:
                data.append({
                    'loan_id': f'NEW_{fid}',
                    'observation_date': date,
                    'score': max(0, min(1, score)),  # Clip to [0,1]
                    'rating': '2',  # Arbitrary valid rating
                    'default_flag': 0,
                    'into_default_flag': 0,
                })
        
        portfolio_df = pd.DataFrame(data)
        
        sim = PortfolioSimulator(
            portfolio_df=portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='observation_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date=datetime(2024, 1, 1),
            asset_correlation=0.15,
            random_seed=42
        )
        sim.prepare_simulation()
        sim._simulate_historical_ratings()
        sim._calculate_migration_matrix()
        sim._calculate_long_term_pd(use_simulated=True)
        
        # Should not crash
        result = sim._simulate_new_clients()
        
        assert 'simulated_rating' in result.columns
        assert result['simulated_rating'].notna().all()
    
    def test_memory_efficient_mode_consistency(self, score_to_rating_bounds):
        """
        Test that memory_efficient mode produces same results as standard mode.
        This mimics integrated_analysis behavior.
        """
        np.random.seed(42)
        dates_hist = pd.date_range('2023-01-31', '2023-12-31', freq='M')
        dates_app = pd.date_range('2024-01-31', '2024-03-31', freq='M')
        
        data = []
        for fid in range(100):
            for date in dates_hist:
                data.append({
                    'loan_id': f'HIST_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3', '4', '5']),
                    'default_flag': 0,
                    'into_default_flag': 0 if np.random.random() > 0.02 else 1,
                })
        
        for fid in range(50):
            for date in dates_app:
                data.append({
                    'loan_id': f'NEW_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3', '4', '5']),
                    'default_flag': 0,
                    'into_default_flag': 0,
                })
        
        portfolio_df = pd.DataFrame(data)
        
        # Standard mode
        sim1 = PortfolioSimulator(
            portfolio_df=portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='observation_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date=datetime(2024, 1, 1),
            asset_correlation=0.15,
            random_seed=42
        )
        result1 = sim1.run_monte_carlo(num_iterations=1, random_seed=42, memory_efficient=False)[0]
        
        # Memory-efficient mode
        sim2 = PortfolioSimulator(
            portfolio_df=portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='observation_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date=datetime(2024, 1, 1),
            asset_correlation=0.15,
            random_seed=42
        )
        result2 = sim2.run_monte_carlo(num_iterations=1, random_seed=42, memory_efficient=True)[0]
        
        # Results should be identical
        new_client_ids = sim1.new_clients_df[sim1.loan_id_col].unique()
        result1_new = result1[result1[sim1.loan_id_col].isin(new_client_ids)].sort_values(
            [sim1.loan_id_col, sim1.date_col]
        ).reset_index(drop=True)
        result2_new = result2[result2[sim2.loan_id_col].isin(new_client_ids)].sort_values(
            [sim2.loan_id_col, sim2.date_col]
        ).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result1_new, result2_new)


class TestRatingBoundaryConditions:
    """Test rating boundary conditions that might cause index errors."""
    
    def test_score_exactly_at_boundary(self, score_to_rating_bounds):
        """Test scores that fall exactly on rating boundaries."""
        # Create scores exactly at boundaries
        boundary_scores = []
        for rating, (lower, upper) in score_to_rating_bounds.items():
            boundary_scores.extend([lower, upper])
        
        # Create portfolio with boundary scores
        np.random.seed(42)
        dates_hist = pd.date_range('2023-01-31', '2023-12-31', freq='M')
        dates_app = pd.date_range('2024-01-31', '2024-03-31', freq='M')
        
        data = []
        for fid in range(100):
            for date in dates_hist:
                data.append({
                    'loan_id': f'HIST_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3']),
                    'default_flag': 0,
                    'into_default_flag': 0,
                })
        
        for fid, score in enumerate(boundary_scores):
            for date in dates_app:
                data.append({
                    'loan_id': f'NEW_{fid}',
                    'observation_date': date,
                    'score': score,
                    'rating': '2',
                    'default_flag': 0,
                    'into_default_flag': 0,
                })
        
        portfolio_df = pd.DataFrame(data)
        
        sim = PortfolioSimulator(
            portfolio_df=portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='observation_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date=datetime(2024, 1, 1),
            asset_correlation=0.15,
            random_seed=42
        )
        sim.prepare_simulation()
        sim._simulate_historical_ratings()
        sim._calculate_migration_matrix()
        sim._calculate_long_term_pd(use_simulated=True)
        
        # Should handle boundary scores gracefully
        result = sim._simulate_new_clients()
        
        assert 'simulated_rating' in result.columns
        assert result['simulated_rating'].notna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=long', '-s'])
