import pytest
import pandas as pd
import numpy as np
from irbstudio.simulation.migration import calculate_migration_matrix

@pytest.fixture
def sample_migration_data():
    """Create a sample DataFrame for testing migration matrix calculation."""
    data = {
        'loan_id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'D', 'D'],
        'date': [
            '2022-01-01', '2022-02-01', '2022-03-01', # A: 1 -> 2 -> 2
            '2022-01-01', '2022-02-01',             # B: 1 -> 1
            '2022-01-01', '2022-02-01', '2022-03-01', # C: 2 -> 3 -> 1
            '2022-01-01', '2022-02-01'              # D: 4 -> 4 (rating 4 is stable)
        ],
        'rating': [1, 2, 2, 1, 1, 2, 3, 1, 4, 4]
    }
    return pd.DataFrame(data)

def test_calculate_migration_matrix_structure(sample_migration_data):
    """Test the basic structure and properties of the output matrix."""
    matrix = calculate_migration_matrix(
        sample_migration_data,
        id_col='loan_id',
        date_col='date',
        rating_col='rating'
    )

    assert isinstance(matrix, pd.DataFrame)
    # All ratings (1, 2, 3, 4) should be present as strings
    assert all(r in matrix.index for r in ['1', '2', '3', '4'])
    # Matrix should be square
    assert matrix.shape == (4, 4)
    # Probabilities should be between 0 and 1
    assert np.all((matrix >= 0) & (matrix <= 1))
    # Rows should sum to 1 (or 0 if no transitions from that rating)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-6)

def test_migration_matrix_values(sample_migration_data):
    """
    Test the calculated probabilities against expected values.
    - From 1: one to 1 (B), one to 2 (A) => 50% to 1, 50% to 2
    - From 2: one to 2 (A), one to 3 (C) => 50% to 2, 50% to 3
    - From 3: one to 1 (C) => 100% to 1
    - From 4: one to 4 (D) => 100% to 4 (stable)
    """
    matrix = calculate_migration_matrix(
        sample_migration_data,
        id_col='loan_id',
        date_col='date',
        rating_col='rating'
    )

    # Expected matrix based on the actual transitions:
    # From 1: one to 1 (B), one to 2 (A) => 50% to 1, 50% to 2
    # From 2: one to 2 (A), one to 3 (C) => 50% to 2, 50% to 3
    # From 3: one to 1 (C) => 100% to 1
    # From 4: one to 4 (D) => 100% to 4 (stable)
    expected = pd.DataFrame(
        [
            [0.5, 0.5, 0.0, 0.0],  # From rating 1: 50% stay at 1, 50% go to 2
            [0.0, 0.5, 0.5, 0.0],  # From rating 2: 50% stay at 2, 50% go to 3
            [1.0, 0.0, 0.0, 0.0],  # From rating 3: 100% go to 1
            [0.0, 0.0, 0.0, 1.0],  # From rating 4: 100% stay at 4
        ],
        index=['1', '2', '3', '4'],    # From ratings
        columns=['1', '2', '3', '4']   # To ratings
    )
    
    pd.testing.assert_frame_equal(matrix, expected)

def test_empty_dataframe():
    """Test with an empty DataFrame."""
    empty_df = pd.DataFrame({'loan_id': [], 'date': [], 'rating': []})
    matrix = calculate_migration_matrix(empty_df, 'loan_id', 'date', 'rating')
    assert matrix.empty

def test_no_transitions_data():
    """Test with data where no loan has more than one observation."""
    data = {
        'loan_id': ['A', 'B', 'C'],
        'date': ['2022-01-01', '2022-01-01', '2022-01-01'],
        'rating': [1, 2, 3]
    }
    df = pd.DataFrame(data)
    matrix = calculate_migration_matrix(df, 'loan_id', 'date', 'rating')

    # Expect an identity matrix, as no transitions occurred.
    expected = pd.DataFrame(np.identity(3), index=['1', '2', '3'], columns=['1', '2', '3'])
    pd.testing.assert_frame_equal(matrix, expected)

def test_missing_column_raises_error(sample_migration_data):
    """Test that a ValueError is raised if a specified column is missing."""
    with pytest.raises(ValueError):
        calculate_migration_matrix(
            sample_migration_data,
            id_col='wrong_id_col',
            date_col='date',
            rating_col='rating'
        )
