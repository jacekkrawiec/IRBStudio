"""
Migration Matrix tests (Priority 2).

This module contains tests for rating migration matrix calculation:
- Basic migration matrix calculation
- Historical transition rates
- Rating grade migrations
- Default transitions
- Validation against historical patterns

Tests verify that migration matrices are calculated correctly from historical data.
"""

import pytest
import numpy as np
import pandas as pd
from irbstudio.simulation.migration import calculate_migration_matrix


class TestMigrationMatrixCalculation:
    """Tests for migration matrix calculation."""
    
    def test_migration_matrix_calculation_basic(self):
        """Test basic migration matrix calculation."""
        # Create simple historical data with known transitions
        data = pd.DataFrame({
            'loan_id': ['L1', 'L1', 'L1', 'L2', 'L2', 'L3', 'L3'],
            'date': pd.to_datetime([
                '2023-01', '2023-02', '2023-03',
                '2023-01', '2023-02',
                '2023-01', '2023-02'
            ]),
            'rating': ['A', 'A', 'B', 'B', 'B', 'A', 'B']
        })
        
        migration_matrix = calculate_migration_matrix(
            data=data,
            id_col='loan_id',
            date_col='date',
            rating_col='rating'
        )
        
        # Verify it's a DataFrame
        assert isinstance(migration_matrix, pd.DataFrame)
        
        # Verify it's square
        assert migration_matrix.shape[0] == migration_matrix.shape[1]
        
        # Verify row sums equal 1 (probabilities)
        row_sums = migration_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
    
    def test_migration_matrix_historical_rates(self):
        """Test migration matrix with historical transition rates."""
        # Create data with predictable transitions
        # A -> A (80%), A -> B (20%)
        # B -> B (70%), B -> C (30%)
        data = pd.DataFrame({
            'loan_id': ['L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L4', 'L4', 'L5', 'L5',
                       'L6', 'L6', 'L7', 'L7', 'L8', 'L8', 'L9', 'L9', 'L10', 'L10'],
            'date': pd.to_datetime([
                '2023-01', '2023-02',  # L1: A->A
                '2023-01', '2023-02',  # L2: A->A
                '2023-01', '2023-02',  # L3: A->A
                '2023-01', '2023-02',  # L4: A->A
                '2023-01', '2023-02',  # L5: A->B
                '2023-01', '2023-02',  # L6: B->B
                '2023-01', '2023-02',  # L7: B->B
                '2023-01', '2023-02',  # L8: B->B
                '2023-01', '2023-02',  # L9: B->B
                '2023-01', '2023-02',  # L10: B->C
            ]),
            'rating': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B',
                      'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'C']
        })
        
        migration_matrix = calculate_migration_matrix(
            data=data,
            id_col='loan_id',
            date_col='date',
            rating_col='rating'
        )
        
        # Check A->A probability (should be ~0.8: 4 out of 5)
        assert migration_matrix.loc['A', 'A'] == pytest.approx(0.8, abs=0.01)
        
        # Check A->B probability (should be ~0.2: 1 out of 5)
        assert migration_matrix.loc['A', 'B'] == pytest.approx(0.2, abs=0.01)
        
        # Check B->B probability (should be ~0.8: 4 out of 5)
        assert migration_matrix.loc['B', 'B'] == pytest.approx(0.8, abs=0.01)
        
        # Check B->C probability (should be ~0.2: 1 out of 5)
        assert migration_matrix.loc['B', 'C'] == pytest.approx(0.2, abs=0.01)
    
    def test_migration_matrix_rating_transitions(self):
        """Test migration matrix captures rating grade migrations."""
        # Create data with upgrades and downgrades
        data = pd.DataFrame({
            'loan_id': ['L1', 'L1', 'L1', 'L2', 'L2', 'L2', 'L3', 'L3', 'L3'],
            'date': pd.to_datetime([
                '2023-01', '2023-02', '2023-03',  # L1: B->A->A (upgrade)
                '2023-01', '2023-02', '2023-03',  # L2: A->B->C (downgrade)
                '2023-01', '2023-02', '2023-03'   # L3: B->B->B (stable)
            ]),
            'rating': ['B', 'A', 'A', 'A', 'B', 'C', 'B', 'B', 'B']
        })
        
        migration_matrix = calculate_migration_matrix(
            data=data,
            id_col='loan_id',
            date_col='date',
            rating_col='rating'
        )
        
        # Verify matrix captures both upgrades and downgrades
        # B->A (upgrade): 1 out of 3 = 33%
        assert migration_matrix.loc['B', 'A'] > 0
        
        # A->B (downgrade): 1 out of 2 = 50%
        assert migration_matrix.loc['A', 'B'] > 0
        
        # B->B (stable): 2 out of 3 = 67%
        assert migration_matrix.loc['B', 'B'] > migration_matrix.loc['B', 'A']
    
    def test_migration_matrix_default_transitions(self):
        """Test migration matrix with default transitions."""
        # Create data including defaults (D rating)
        data = pd.DataFrame({
            'loan_id': ['L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L4', 'L4'],
            'date': pd.to_datetime([
                '2023-01', '2023-02',  # L1: A->A (no default)
                '2023-01', '2023-02',  # L2: B->D (default)
                '2023-01', '2023-02',  # L3: C->D (default)
                '2023-01', '2023-02'   # L4: B->B (no default)
            ]),
            'rating': ['A', 'A', 'B', 'D', 'C', 'D', 'B', 'B']
        })
        
        migration_matrix = calculate_migration_matrix(
            data=data,
            id_col='loan_id',
            date_col='date',
            rating_col='rating'
        )
        
        # Verify default transitions are captured
        # B->D: 1 out of 2 = 50%
        assert migration_matrix.loc['B', 'D'] == pytest.approx(0.5, abs=0.01)
        
        # C->D: 1 out of 1 = 100%
        assert migration_matrix.loc['C', 'D'] == pytest.approx(1.0, abs=0.01)
    
    def test_migration_matrix_stable_state(self):
        """Test migration matrix with mostly stable ratings."""
        # Create data where most loans stay in same rating
        np.random.seed(42)
        loan_ids = []
        dates = []
        ratings = []
        
        for loan_id in range(1, 21):  # 20 loans
            for month in range(1, 7):  # 6 months
                loan_ids.append(f'L{loan_id}')
                dates.append(f'2023-{month:02d}')
                
                # 90% stay in same rating, 10% change
                if loan_id <= 18:  # 18 loans stay as 'A'
                    ratings.append('A')
                else:  # 2 loans transition A->B
                    ratings.append('A' if month <= 3 else 'B')
        
        data = pd.DataFrame({
            'loan_id': loan_ids,
            'date': pd.to_datetime(dates),
            'rating': ratings
        })
        
        migration_matrix = calculate_migration_matrix(
            data=data,
            id_col='loan_id',
            date_col='date',
            rating_col='rating'
        )
        
        # A->A probability should be very high (stable)
        assert migration_matrix.loc['A', 'A'] > 0.85
    
    def test_migration_matrix_validation(self):
        """Test migration matrix validation against historical patterns."""
        # Create realistic migration data
        # AAA: 95% stable, 5% downgrade to AA
        # AA: 90% stable, 8% downgrade to A, 2% upgrade to AAA
        # A: 85% stable, 10% downgrade to BBB, 5% upgrade to AA
        data = []
        
        for i in range(20):
            # AAA loans
            data.append({'loan_id': f'AAA_{i}', 'date': '2023-01', 'rating': 'AAA'})
            if i < 19:
                data.append({'loan_id': f'AAA_{i}', 'date': '2023-02', 'rating': 'AAA'})
            else:
                data.append({'loan_id': f'AAA_{i}', 'date': '2023-02', 'rating': 'AA'})
        
        for i in range(20):
            # AA loans
            data.append({'loan_id': f'AA_{i}', 'date': '2023-01', 'rating': 'AA'})
            if i < 18:
                data.append({'loan_id': f'AA_{i}', 'date': '2023-02', 'rating': 'AA'})
            elif i < 19:
                data.append({'loan_id': f'AA_{i}', 'date': '2023-02', 'rating': 'A'})
            else:
                data.append({'loan_id': f'AA_{i}', 'date': '2023-02', 'rating': 'AAA'})
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        migration_matrix = calculate_migration_matrix(
            data=df,
            id_col='loan_id',
            date_col='date',
            rating_col='rating'
        )
        
        # Validate diagonal dominance (most loans stay in same rating)
        assert migration_matrix.loc['AAA', 'AAA'] > 0.9
        assert migration_matrix.loc['AA', 'AA'] > 0.85
        
        # Validate transition patterns
        assert migration_matrix.loc['AAA', 'AA'] < 0.1
        assert migration_matrix.loc['AA', 'A'] < 0.15


class TestMigrationMatrixEdgeCases:
    """Tests for migration matrix edge cases."""
    
    def test_migration_matrix_single_rating(self):
        """Test migration matrix with single rating only."""
        data = pd.DataFrame({
            'loan_id': ['L1', 'L1', 'L2', 'L2', 'L3', 'L3'],
            'date': pd.to_datetime([
                '2023-01', '2023-02',
                '2023-01', '2023-02',
                '2023-01', '2023-02'
            ]),
            'rating': ['A', 'A', 'A', 'A', 'A', 'A']  # All same rating
        })
        
        migration_matrix = calculate_migration_matrix(
            data=data,
            id_col='loan_id',
            date_col='date',
            rating_col='rating'
        )
        
        # Should have 100% probability of staying in same rating
        assert migration_matrix.loc['A', 'A'] == pytest.approx(1.0, abs=1e-6)
    
    def test_migration_matrix_missing_columns(self):
        """Test migration matrix with missing required columns."""
        data = pd.DataFrame({
            'loan_id': ['L1', 'L1'],
            'date': pd.to_datetime(['2023-01', '2023-02'])
            # Missing 'rating' column
        })
        
        with pytest.raises(ValueError, match="specified columns are not in"):
            calculate_migration_matrix(
                data=data,
                id_col='loan_id',
                date_col='date',
                rating_col='rating'
            )


# Summary of test coverage:
# - Basic migration matrix: 1 test
# - Historical rates: 1 test
# - Rating transitions: 1 test
# - Default transitions: 1 test
# - Stable state: 1 test
# - Validation: 1 test
# - Edge cases: 2 tests
# Total: 8 migration matrix tests
