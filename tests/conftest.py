"""
Pytest configuration and fixtures for IRBStudio tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import yaml


# Test data paths
TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent
DATA_DIR = PROJECT_DIR / "data"


@pytest.fixture
def sample_portfolio_df():
    """Create a sample portfolio DataFrame for testing."""
    np.random.seed(42)
    n_loans = 1000
    
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='Q')
    
    data = {
        'loan_id': [f'LOAN_{i:06d}' for i in range(n_loans)],
        'exposure': np.random.uniform(100000, 1000000, n_loans),
        'pd': np.random.uniform(0.001, 0.15, n_loans),
        'score': np.random.uniform(300, 850, n_loans),
        'rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'], n_loans),
        'reporting_date': np.random.choice(dates, n_loans),
        'default_flag': np.random.choice([0, 1], n_loans, p=[0.97, 0.03]),
        'into_default_flag': np.random.choice([0, 1], n_loans, p=[0.98, 0.02]),
        'lgd': np.random.uniform(0.15, 0.35, n_loans),
        'maturity': np.random.uniform(1, 30, n_loans),
        'property_value': np.random.uniform(150000, 1500000, n_loans),
        'ltv': np.random.uniform(0.3, 0.95, n_loans),
        'segment': np.random.choice(['Prime', 'Near-Prime', 'Subprime'], n_loans),
        'product_type': np.random.choice(['Fixed', 'Variable', 'Interest-Only'], n_loans),
    }
    
    df = pd.DataFrame(data)
    df['reporting_date'] = pd.to_datetime(df['reporting_date'])
    
    return df


@pytest.fixture
def small_portfolio_df():
    """Create a small portfolio (100 loans) for quick tests with date range for simulation."""
    np.random.seed(42)
    n_loans = 100
    
    # Create dates spanning 18 months to allow for historical/application split
    dates = pd.date_range('2023-07-01', '2024-12-31', freq='M')
    
    data = {
        'loan_id': [f'LOAN_{i:06d}' for i in range(n_loans)],
        'exposure': np.random.uniform(100000, 500000, n_loans),
        'pd': np.random.uniform(0.01, 0.1, n_loans),
        'score': np.random.uniform(300, 750, n_loans),
        'rating': np.random.choice(['A', 'BBB', 'BB'], n_loans),
        'reporting_date': np.random.choice(dates, n_loans),
        'default_flag': np.random.choice([0, 1], n_loans, p=[0.95, 0.05]),
        'into_default_flag': np.random.choice([0, 1], n_loans, p=[0.98, 0.02]),
        'lgd': 0.25,
        'maturity': 5.0,
        'property_value': np.random.uniform(150000, 750000, n_loans),
        'ltv': np.random.uniform(0.4, 0.8, n_loans),
    }
    
    df = pd.DataFrame(data)
    df['reporting_date'] = pd.to_datetime(df['reporting_date'])
    
    return df


@pytest.fixture
def multi_date_portfolio_df():
    """Create a portfolio with multiple reporting dates."""
    np.random.seed(42)
    n_loans_per_date = 100
    dates = pd.date_range('2024-01-31', '2024-12-31', freq='M')
    
    dfs = []
    for date in dates:
        data = {
            'loan_id': [f'LOAN_{i:06d}' for i in range(n_loans_per_date)],
            'exposure': np.random.uniform(100000, 500000, n_loans_per_date),
            'pd': np.random.uniform(0.01, 0.1, n_loans_per_date),
            'rating': np.random.choice(['A', 'BBB', 'BB'], n_loans_per_date),
            'reporting_date': date,
            'default_flag': np.random.choice([0, 1], n_loans_per_date, p=[0.95, 0.05]),
            'into_default_flag': np.random.choice([0, 1], n_loans_per_date, p=[0.98, 0.02]),
            'lgd': 0.25,
            'maturity': 5.0,
            'property_value': np.random.uniform(150000, 750000, n_loans_per_date),
            'ltv': np.random.uniform(0.4, 0.8, n_loans_per_date),
        }
        df = pd.DataFrame(data)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['reporting_date'] = pd.to_datetime(combined_df['reporting_date'])
    
    return combined_df


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        'column_mapping': {
            'loan_id': 'loan_id',
            'exposure': 'exposure',
            'ltv': 'ltv',
            'rating': 'rating',
            'date': 'reporting_date',
            'default_flag': 'default_flag',
            'into_default_flag': 'into_default_flag',
        },
        'regulatory': {
            'jurisdiction': 'generic',
            'asset_correlation': 0.15,
            'confidence_level': 0.999,
        },
        'scenarios': [
            {
                'name': 'Baseline',
                'description': 'Current model performance',
                'pd_auc': 0.80,
                'portfolio_default_rate': 0.03,
                'lgd': 0.25,
                'new_loan_rate': 0.0,
                'rating_pd_map': {
                    'AAA': 0.001,
                    'AA': 0.005,
                    'A': 0.01,
                    'BBB': 0.03,
                    'BB': 0.05,
                    'B': 0.10,
                },
            }
        ],
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Create a temporary YAML config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config_dict, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_csv_file(small_portfolio_df):
    """Create a temporary CSV file with portfolio data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        small_portfolio_df.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def airb_params():
    """Standard AIRB parameters for testing."""
    return {
        'asset_correlation': 0.15,
        'confidence_level': 0.999,
        'lgd': 0.25,
        'maturity_adjustment': False,
    }


@pytest.fixture
def sa_params():
    """Standard SA parameters for testing."""
    return {
        'secured_portion_rw': 0.20,
        'unsecured_portion_rw': 0.75,
        'property_value_threshold': 0.55,
    }


@pytest.fixture
def column_mapping():
    """Standard column mapping for testing."""
    return {
        'loan_id': 'loan_id',
        'exposure': 'exposure',
        'pd': 'pd',
        'rating': 'rating',
        'date': 'reporting_date',
        'default_flag': 'default_flag',
        'into_default_flag': 'into_default_flag',
    }


@pytest.fixture
def score_to_rating_bounds():
    """Define rating bounds for score-to-rating mapping."""
    return {
        'AAA': (0.0, 0.01),
        'AA': (0.01, 0.03),
        'A': (0.03, 0.05),
        'BBB': (0.05, 0.10),
        'BB': (0.10, 0.15),
        'B': (0.15, 1.0),
        'D': (0.0, 0.0),  # Default rating handled separately
    }


# Helper functions for tests
def assert_dataframe_structure(df, required_columns):
    """Assert that DataFrame has required columns."""
    assert isinstance(df, pd.DataFrame), "Result must be a DataFrame"
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"


def assert_numeric_column(df, column_name):
    """Assert that column contains numeric values."""
    assert column_name in df.columns, f"Column {column_name} not found"
    assert pd.api.types.is_numeric_dtype(df[column_name]), \
        f"Column {column_name} must be numeric"


def assert_positive_values(df, column_name):
    """Assert that column contains only positive values."""
    assert_numeric_column(df, column_name)
    assert (df[column_name] >= 0).all(), \
        f"Column {column_name} must contain only non-negative values"


def assert_in_range(value, min_val, max_val, name="Value"):
    """Assert that value is within range."""
    assert min_val <= value <= max_val, \
        f"{name} must be between {min_val} and {max_val}, got {value}"
