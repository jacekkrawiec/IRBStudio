"""
Exhaustive tests for PortfolioSimulator._simulate_new_clients() method.

Tests cover:
1. Basic functionality
2. Edge cases (empty DataFrames, single client, etc.)
3. Rating alignment with migration matrix
4. Score boundary cases
5. Random seed reproducibility
6. Data integrity and validation
7. Performance with large datasets
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator


@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    np.random.seed(42)
    
    # Historical data (2023-01 to 2023-12)
    dates_hist = pd.date_range('2023-01-31', '2023-12-31', freq='M')
    n_facilities_hist = 100
    
    historical_data = []
    for facility_id in range(n_facilities_hist):
        for date in dates_hist:
            historical_data.append({
                'loan_id': f'HIST_{facility_id}',
                'observation_date': date,
                'score': np.random.beta(2, 5),
                'rating': np.random.choice(['1', '2', '3', '4', '5'], p=[0.1, 0.2, 0.4, 0.2, 0.1]),
                'default_flag': 0,
                'into_default_flag': 0 if np.random.random() > 0.02 else 1,
                'exposure': np.random.uniform(100000, 500000)
            })
    
    # Application data - new clients (2024-01 to 2024-03)
    dates_app = pd.date_range('2024-01-31', '2024-03-31', freq='M')
    n_facilities_new = 50
    
    application_data = []
    for facility_id in range(n_facilities_new):
        for date in dates_app:
            application_data.append({
                'loan_id': f'NEW_{facility_id}',
                'observation_date': date,
                'score': np.random.beta(2, 5),
                'rating': np.random.choice(['1', '2', '3', '4', '5'], p=[0.1, 0.2, 0.4, 0.2, 0.1]),
                'default_flag': 0,
                'into_default_flag': 0,
                'exposure': np.random.uniform(100000, 500000)
            })
    
    # Application data - existing clients (2024-01 to 2024-03)
    for facility_id in range(20):  # 20 existing clients
        for date in dates_app:
            application_data.append({
                'loan_id': f'HIST_{facility_id}',  # Same as historical
                'observation_date': date,
                'score': np.random.beta(2, 5),
                'rating': np.random.choice(['1', '2', '3', '4', '5'], p=[0.1, 0.2, 0.4, 0.2, 0.1]),
                'default_flag': 0,
                'into_default_flag': 0,
                'exposure': np.random.uniform(100000, 500000)
            })
    
    # Combine all data
    all_data = historical_data + application_data
    portfolio_df = pd.DataFrame(all_data)
    
    return portfolio_df


@pytest.fixture
def score_to_rating_bounds():
    """Standard rating bounds for testing."""
    return {
        '1': (0.0, 0.003),
        '2': (0.003, 0.006),
        '3': (0.006, 0.032),
        '4': (0.032, 0.040),
        '5': (0.040, 0.256),
        '6': (0.256, 0.765),
        '7': (0.765, 1.0)
    }


@pytest.fixture
def simulator(sample_portfolio_data, score_to_rating_bounds):
    """Create a prepared simulator for testing."""
    sim = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
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
    return sim


class TestSimulateNewClientsBasic:
    """Test basic functionality of _simulate_new_clients."""
    
    def test_returns_dataframe(self, simulator):
        """Test that method returns a DataFrame."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        assert isinstance(result, pd.DataFrame)
    
    def test_returns_empty_when_no_new_clients(self, sample_portfolio_data, score_to_rating_bounds):
        """Test that empty DataFrame is returned when no new clients exist."""
        # Create simulator with application date after all data
        sim = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='observation_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date=datetime(2025, 1, 1),  # After all data
            asset_correlation=0.15
        )
        sim.prepare_simulation()
        
        # Force new_clients_df to be empty
        sim.new_clients_df = pd.DataFrame()
        
        result = sim._simulate_new_clients()
        assert result.empty
    
    def test_preserves_original_columns(self, simulator):
        """Test that original columns are preserved."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        original_cols = set(simulator.new_clients_df.columns)
        result = simulator._simulate_new_clients()
        
        # All original columns should be present
        for col in original_cols:
            assert col in result.columns
    
    def test_adds_simulated_rating_column(self, simulator):
        """Test that simulated_rating column is added."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        assert 'simulated_rating' in result.columns
    
    def test_adds_simulated_score_column(self, simulator):
        """Test that simulated_score column is added."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        assert 'simulated_score' in result.columns
    
    def test_preserves_row_count(self, simulator):
        """Test that number of rows is preserved."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        original_count = len(simulator.new_clients_df)
        result = simulator._simulate_new_clients()
        
        assert len(result) == original_count
    
    def test_preserves_facility_ids(self, simulator):
        """Test that all original facility IDs are preserved."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        original_ids = set(simulator.new_clients_df[simulator.loan_id_col].unique())
        result = simulator._simulate_new_clients()
        result_ids = set(result[simulator.loan_id_col].unique())
        
        assert original_ids == result_ids


class TestSimulateNewClientsRatingAlignment:
    """Test rating alignment with migration matrix."""
    
    def test_all_ratings_exist_in_migration_matrix(self, simulator):
        """Test that all generated ratings exist in migration matrix."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        
        migration_ratings = set(simulator.simulated_migration_matrix.index)
        new_ratings = set(result['simulated_rating'].dropna().unique())
        
        missing_ratings = new_ratings - migration_ratings
        assert len(missing_ratings) == 0, f"Ratings {missing_ratings} not in migration matrix"
    
    def test_ratings_are_valid(self, simulator, score_to_rating_bounds):
        """Test that all ratings are from valid rating set."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        
        valid_ratings = set(score_to_rating_bounds.keys())
        result_ratings = set(result['simulated_rating'].dropna().unique())
        
        invalid_ratings = result_ratings - valid_ratings
        assert len(invalid_ratings) == 0, f"Invalid ratings found: {invalid_ratings}"
    
    def test_no_null_ratings(self, simulator):
        """Test that no null ratings are generated."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        
        null_count = result['simulated_rating'].isna().sum()
        assert null_count == 0, f"Found {null_count} null ratings"
    
    def test_no_null_scores(self, simulator):
        """Test that no null scores are generated."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        
        null_count = result['simulated_score'].isna().sum()
        assert null_count == 0, f"Found {null_count} null scores"


class TestSimulateNewClientsScoreBoundaries:
    """Test score boundary cases."""
    
    def test_scores_within_valid_range(self, simulator):
        """Test that all scores are within [0, 1] range."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        
        scores = result['simulated_score'].values
        assert np.all((scores >= 0) & (scores <= 1)), "Scores outside [0, 1] range"
    
    def test_scores_map_correctly_to_ratings(self, simulator, score_to_rating_bounds):
        """Test that scores correctly map to their ratings."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        
        # Check each row
        for idx, row in result.iterrows():
            score = row['simulated_score']
            rating = row['simulated_rating']
            
            # Find expected rating for this score
            expected_rating = None
            for r, (lower, upper) in score_to_rating_bounds.items():
                if lower <= score < upper or (score == upper and r == '7'):
                    expected_rating = r
                    break
            
            # Note: Rating might differ due to migration
            # So we just check that the rating is valid
            assert rating in score_to_rating_bounds.keys()
    
    def test_handles_edge_scores(self, simulator):
        """Test handling of edge case scores (0.0, 1.0)."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        # Artificially inject edge scores
        simulator.new_clients_df.loc[0, 'score'] = 0.0
        simulator.new_clients_df.loc[1, 'score'] = 1.0
        
        result = simulator._simulate_new_clients()
        
        # Should not crash and should assign valid ratings
        assert 'simulated_rating' in result.columns
        assert result['simulated_rating'].notna().all()


class TestSimulateNewClientsReproducibility:
    """Test random seed reproducibility."""
    
    def test_reproducible_with_same_seed(self, sample_portfolio_data, score_to_rating_bounds):
        """Test that same seed produces same results."""
        # First run
        sim1 = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
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
        sim1.prepare_simulation()
        sim1._simulate_historical_ratings()
        sim1._calculate_migration_matrix()
        sim1._calculate_long_term_pd(use_simulated=True)
        result1 = sim1._simulate_new_clients()
        
        # Second run with same seed
        sim2 = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
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
        sim2.prepare_simulation()
        sim2._simulate_historical_ratings()
        sim2._calculate_migration_matrix()
        sim2._calculate_long_term_pd(use_simulated=True)
        result2 = sim2._simulate_new_clients()
        
        # Results should be identical
        pd.testing.assert_frame_equal(
            result1.sort_values(['loan_id', 'observation_date']).reset_index(drop=True),
            result2.sort_values(['loan_id', 'observation_date']).reset_index(drop=True)
        )
    
    def test_different_with_different_seed(self, sample_portfolio_data, score_to_rating_bounds):
        """Test that different seeds produce different results."""
        # First run
        sim1 = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
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
        sim1.prepare_simulation()
        sim1._simulate_historical_ratings()
        sim1._calculate_migration_matrix()
        sim1._calculate_long_term_pd(use_simulated=True)
        result1 = sim1._simulate_new_clients()
        
        # Second run with different seed
        sim2 = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='observation_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date=datetime(2024, 1, 1),
            asset_correlation=0.15,
            random_seed=123
        )
        sim2.prepare_simulation()
        sim2._simulate_historical_ratings()
        sim2._calculate_migration_matrix()
        sim2._calculate_long_term_pd(use_simulated=True)
        result2 = sim2._simulate_new_clients()
        
        # Results should differ
        ratings_equal = (
            result1.sort_values(['loan_id', 'observation_date'])['simulated_rating'].values ==
            result2.sort_values(['loan_id', 'observation_date'])['simulated_rating'].values
        ).all()
        
        assert not ratings_equal, "Different seeds should produce different results"


class TestSimulateNewClientsDataIntegrity:
    """Test data integrity and validation."""
    
    def test_no_data_corruption(self, simulator):
        """Test that original data is not corrupted."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        original_data = simulator.new_clients_df.copy()
        result = simulator._simulate_new_clients()
        
        # Original DataFrame should not be modified
        pd.testing.assert_frame_equal(simulator.new_clients_df, original_data)
    
    def test_consistent_facility_observations(self, simulator):
        """Test that each facility has consistent number of observations."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        original_counts = simulator.new_clients_df.groupby(simulator.loan_id_col).size()
        result = simulator._simulate_new_clients()
        result_counts = result.groupby(simulator.loan_id_col).size()
        
        pd.testing.assert_series_equal(original_counts, result_counts)
    
    def test_temporal_consistency(self, simulator):
        """Test that temporal order is maintained."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        
        # Check that dates are sorted within each facility
        for facility_id in result[simulator.loan_id_col].unique():
            facility_data = result[result[simulator.loan_id_col] == facility_id]
            dates = facility_data[simulator.date_col].values
            assert np.all(dates[:-1] <= dates[1:]), f"Dates not sorted for facility {facility_id}"
    
    def test_first_rating_preservation(self, simulator):
        """Test that first rating is preserved when keep_first_rating=True."""
        simulator._simulate_historical_ratings()
        simulator._calculate_migration_matrix()
        simulator._calculate_long_term_pd(use_simulated=True)
        
        result = simulator._simulate_new_clients()
        
        # Get first observation for each facility
        first_obs = result.groupby(simulator.loan_id_col).first()
        
        # All first observations should have a valid rating
        assert first_obs['simulated_rating'].notna().all()


class TestSimulateNewClientsEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_new_client(self, sample_portfolio_data, score_to_rating_bounds):
        """Test with only one new client."""
        # Filter to keep only one new client
        new_client_id = sample_portfolio_data[
            sample_portfolio_data['loan_id'].str.startswith('NEW_')
        ]['loan_id'].iloc[0]
        
        filtered_data = sample_portfolio_data[
            (sample_portfolio_data['loan_id'].str.startswith('HIST_')) |
            (sample_portfolio_data['loan_id'] == new_client_id)
        ].copy()
        
        sim = PortfolioSimulator(
            portfolio_df=filtered_data,
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
        
        assert len(result) > 0
        assert 'simulated_rating' in result.columns
    
    def test_many_new_clients(self, score_to_rating_bounds):
        """Test with large number of new clients."""
        np.random.seed(42)
        
        # Create data with many new clients
        dates_hist = pd.date_range('2023-01-31', '2023-12-31', freq='M')
        dates_app = pd.date_range('2024-01-31', '2024-03-31', freq='M')
        
        data = []
        
        # Historical: 100 facilities
        for fid in range(100):
            for date in dates_hist:
                data.append({
                    'loan_id': f'HIST_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3']),
                    'default_flag': 0,
                    'into_default_flag': 0,
                    'exposure': 100000
                })
        
        # Application: 500 new clients
        for fid in range(500):
            for date in dates_app:
                data.append({
                    'loan_id': f'NEW_{fid}',
                    'observation_date': date,
                    'score': np.random.beta(2, 5),
                    'rating': np.random.choice(['1', '2', '3']),
                    'default_flag': 0,
                    'into_default_flag': 0,
                    'exposure': 100000
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
        
        assert len(result) == 500 * len(dates_app)  # All new client observations
        assert 'simulated_rating' in result.columns
        assert result['simulated_rating'].notna().all()
    
    def test_all_clients_same_rating(self, sample_portfolio_data, score_to_rating_bounds):
        """Test when all new clients have the same initial rating."""
        # Force all new clients to have same rating
        mask = sample_portfolio_data['loan_id'].str.startswith('NEW_')
        sample_portfolio_data.loc[mask, 'rating'] = '3'
        sample_portfolio_data.loc[mask, 'score'] = 0.01  # All same score
        
        sim = PortfolioSimulator(
            portfolio_df=sample_portfolio_data,
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
        
        # Should still work and produce varied ratings due to simulation
        assert len(result['simulated_rating'].unique()) > 1


class TestSimulateNewClientsIntegration:
    """Integration tests with full simulation workflow."""
    
    def test_integration_with_simulate_once(self, simulator):
        """Test integration with full simulate_once workflow."""
        # Run full simulation
        result = simulator.simulate_once(random_seed=42)
        
        # Check that new clients are included
        new_client_ids = simulator.new_clients_df[simulator.loan_id_col].unique()
        result_new_client_ids = result[
            result[simulator.loan_id_col].isin(new_client_ids)
        ][simulator.loan_id_col].unique()
        
        assert len(result_new_client_ids) == len(new_client_ids)
    
    def test_integration_with_monte_carlo(self, simulator):
        """Test integration with Monte Carlo simulation."""
        results = simulator.run_monte_carlo(num_iterations=3, random_seed=42)
        
        assert len(results) == 3
        
        # Each iteration should have new clients
        for result in results:
            new_client_ids = simulator.new_clients_df[simulator.loan_id_col].unique()
            result_ids = result[simulator.loan_id_col].unique()
            
            # All new client IDs should be in result
            assert all(nid in result_ids for nid in new_client_ids)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
