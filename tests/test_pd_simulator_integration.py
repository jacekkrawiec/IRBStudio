import pandas as pd
import pytest
import numpy as np
from irbstudio.simulation.pd_simulator import simulate_portfolio

@pytest.fixture(scope="module")
def sample_config():
    """Sample configuration for portfolio simulation."""
    return {
        'column_mapping': {
            'loan_id_col': 'loan_id',
            'date_col': 'date',
            'rating_col': 'rating',
            'pd_col': 'pd',
            'into_default_flag_col': 'is_default'
        },
        'scenario': {
            'target_auc': 0.75,
            'rating_pd_map': {
                1: 0.001, 2: 0.002, 3: 0.005, 4: 0.01,
                5: 0.02, 6: 0.05, 7: 0.1, 8: 0.15, 'D': 1.0
            },
            'score_to_rating_bounds': {
                1: (0, 0.0015), 
                2: (0.0015, 0.0035), 
                3: (0.0035, 0.0075), 
                4: (0.0075, 0.015), 
                5: (0.015, 0.05), 
                6: (0.05, 0.1),
                7: (0.1, 0.15),
                8: (0.15, 1.0)
            }
        },
        'regulatory_params': {
            'asset_correlation': 0.15
        }
    }

@pytest.fixture(scope="module")
def sample_data():
    """Generate realistic sample data for testing."""
    N_LOANS = 500  # Increased sample size for better fitting
    N_EXISTING = 400
    DEFAULT_RATE = 0.05
    START_YEAR = 2020
    END_YEAR = 2024
    APP_START_YEAR = 2023

    # Generate loan IDs and dates
    loan_ids = [f'L{i:03d}' for i in range(N_LOANS)]
    dates = pd.to_datetime([f'{year}-12-31' for year in range(START_YEAR, END_YEAR + 1)])
    
    # Create DataFrame with all combinations of loan_ids and dates
    df = pd.MultiIndex.from_product(
        [loan_ids, dates], names=['loan_id', 'date']
    ).to_frame(index=False)

    # Segment into existing and new loans
    existing_ids = loan_ids[:N_EXISTING]
    new_ids = loan_ids[N_EXISTING:]
    
    # New loans only appear in the application period
    df = df[
        (df['loan_id'].isin(existing_ids)) | 
        (df['loan_id'].isin(new_ids) & (df['date'].dt.year >= APP_START_YEAR))
    ]

    # Generate ratings
    np.random.seed(42)
    start_ratings = pd.Series(np.random.randint(1, 7, size=N_LOANS), index=loan_ids)
    df['start_rating'] = df['loan_id'].map(start_ratings)

    # Simulate rating migration as a random walk
    years = df['date'].dt.year - START_YEAR
    rating_changes = np.random.randint(-1, 2, size=len(df))
    df['rating'] = df['start_rating'] + years * rating_changes
    df['rating'] = df['rating'].clip(1, 8)

    # Generate defaults
    default_loan_ids = np.random.choice(loan_ids, size=int(N_LOANS * DEFAULT_RATE), replace=False)
    default_dates = {
        loan_id: np.random.choice(dates[dates.year >= START_YEAR + 1])
        for loan_id in default_loan_ids
    }

    df['is_default'] = 0
    for loan_id, default_date in default_dates.items():
        # Mark the first default observation
        default_idx = df[(df['loan_id'] == loan_id) & (df['date'] == default_date)].index
        if not default_idx.empty:
            df.loc[default_idx[0], 'is_default'] = 1
        
        # Mark all subsequent observations as defaulted (rating 'D')
        post_default = (df['loan_id'] == loan_id) & (df['date'] >= default_date)
        df.loc[post_default, 'rating'] = 'D'

    # Generate PDs based on ratings with more diversity for better beta fitting
    # Generate more diverse PDs using a beta distribution for each rating
    np.random.seed(123)
    
    # Create a more diverse set of PDs for each rating
    rating_to_pd_mean = {
        1: 0.001, 2: 0.002, 3: 0.005, 4: 0.01,
        5: 0.02, 6: 0.05, 7: 0.1, 8: 0.15, 'D': 1.0
    }
    
    df['pd'] = np.nan
    
    # For each rating (except 'D'), generate PDs from a beta distribution
    for rating in range(1, 9):
        mask = df['rating'] == rating
        if not mask.any():
            continue
            
        mean_pd = rating_to_pd_mean[rating]
        # Use beta distribution parameters that maintain the desired mean
        # but provide good spread
        if mean_pd < 0.01:
            # For very low PDs, use parameters that give a right-skewed distribution
            a, b = 2, 2/mean_pd - 2
        elif mean_pd < 0.1:
            # For medium-low PDs
            a, b = 3, 3/mean_pd - 3
        else:
            # For higher PDs
            a, b = 5, 5/mean_pd - 5
            
        # Generate PDs from beta distribution
        count = mask.sum()
        df.loc[mask, 'pd'] = np.clip(
            np.random.beta(a, b, size=count),
            0.001,  # Minimum PD 
            0.999   # Maximum PD
        )
    
    # Set PD for defaulted ratings
    df.loc[df['rating'] == 'D', 'pd'] = 1.0
    df.loc[df['is_default'] == 1, 'pd'] = 1.0

    return df.drop(columns=['start_rating'])

def test_simulate_portfolio_integration(sample_config, sample_data):
    """Integration test for the end-to-end portfolio simulation."""
    # Define application period
    application_start_date = pd.to_datetime('2023-12-31')
    
    # Extract column names from config for easier reference
    loan_id_col = sample_config['column_mapping']['loan_id_col']
    date_col = sample_config['column_mapping']['date_col']
    rating_col = sample_config['column_mapping']['rating_col']
    pd_col = sample_config['column_mapping']['pd_col']
    default_flag_col = sample_config['column_mapping']['into_default_flag_col']
    
    # Get count of defaulted loans for later verification
    defaulted_loans = sample_data[sample_data[default_flag_col] == 1][loan_id_col].unique()
    default_dates = {}
    for loan in defaulted_loans:
        default_dates[loan] = sample_data[
            (sample_data[loan_id_col] == loan) & 
            (sample_data[default_flag_col] == 1)
        ][date_col].min()
    
    # Run the simulation
    simulated_df = simulate_portfolio(
        portfolio_df=sample_data,
        application_start_date=application_start_date,
        loan_id_col=loan_id_col,
        date_col=date_col,
        rating_col=rating_col,
        score_col=pd_col,
        into_default_flag_col=default_flag_col,
        target_auc=sample_config['scenario']['target_auc'],
        rating_pd_map=sample_config['scenario']['rating_pd_map'],
        score_to_rating_bounds=sample_config['scenario']['score_to_rating_bounds'],
        asset_correlation=sample_config['regulatory_params']['asset_correlation'],
        default_col=default_flag_col,
    )
    
    # --- Assertions ---
    # Basic structural checks
    assert isinstance(simulated_df, pd.DataFrame)
    assert not simulated_df.empty
    assert 'simulated_pd' in simulated_df.columns
    assert 'simulated_rating' in simulated_df.columns
    
    # Check that the output has the same number of rows as the application sample
    assert len(simulated_df) == len(sample_data), \
        f"Expected {len(sample_data)} rows, got {len(simulated_df)}"

    # Check that simulated ratings and PDs are properly assigned for defaulted loans
    defaulted_rows = simulated_df.loc[simulated_df[default_flag_col] == 1]
    if not defaulted_rows.empty:
        # All rows from default date onward should have PD=1.0 and rating='D'
        assert (defaulted_rows['simulated_pd'] == 1.0).all(), \
            f"Not all defaulted rows have simulated_pd=1.0"
        assert (defaulted_rows['simulated_rating'] == 'D').all(), \
            f"Not all defaulted rows have simulated_rating='D'"
    
    # Check that all PDs are plausible
    assert simulated_df['simulated_pd'].between(0, 1).all(), \
        "Some simulated PDs are outside the valid range [0,1]"
    assert simulated_df['simulated_pd'].isnull().sum() == 0, \
        "Found null values in simulated_pd"
    assert simulated_df['simulated_rating'].isnull().sum() == 0, \
        "Found null values in simulated_rating"
    
    # Check that the final PDs correspond to the rating_pd_map for non-defaults
    non_defaults = simulated_df[simulated_df['simulated_rating'] != 'D']
    if not non_defaults.empty:
        expected_pds = non_defaults['simulated_rating'].map(sample_config['scenario']['rating_pd_map'])
        pd.testing.assert_series_equal(
            non_defaults['simulated_pd'],
            expected_pds.reset_index(drop=True),
            check_names=False
        )
    
    # Check that PDs for defaulted loans are 1.0
    defaults = simulated_df[simulated_df['simulated_rating'] == 'D']
    if not defaults.empty:
        assert (defaults['simulated_pd'] == 1.0).all(), \
            "Found defaulted loans with simulated_pd != 1.0"

def test_simulate_portfolio_no_defaults(sample_config):
    """Test simulation with a dataset that has no defaults."""
    # Create a historical dataset for migration matrix calculation
    np.random.seed(42)
    loan_ids = [f'L{i:03d}' for i in range(50)]  # More loans
    hist_dates = pd.to_datetime(['2020-12-31', '2021-12-31', '2022-12-31'])
    app_date = pd.to_datetime('2023-12-31')
    
    # Create historical data
    hist_df = pd.MultiIndex.from_product(
        [loan_ids, hist_dates], names=['loan_id', 'date']
    ).to_frame(index=False)
    
    # Generate historical ratings with some patterns
    hist_df['rating'] = np.random.randint(1, 9, size=len(hist_df))
    hist_df['is_default'] = 0
    
    # Generate PDs for historical data
    hist_df['pd'] = np.clip(np.random.beta(2, 20, size=len(hist_df)), 0.001, 0.999)
    
    # Create application data (single date)
    app_df = pd.DataFrame({
        'loan_id': loan_ids,
        'date': app_date,
        'rating': np.random.randint(1, 9, size=len(loan_ids)),
        'is_default': 0,
        'pd': np.clip(np.random.beta(2, 20, size=len(loan_ids)), 0.001, 0.999)
    })
    
    # Combine historical and application data
    df = pd.concat([hist_df, app_df], ignore_index=True)
    
    # Run the simulation
    simulated_df = simulate_portfolio(
        portfolio_df=df,
        application_start_date=app_date,
        loan_id_col='loan_id',
        date_col='date',
        rating_col='rating',
        score_col='pd',
        into_default_flag_col='is_default',
        target_auc=sample_config['scenario']['target_auc'],
        rating_pd_map=sample_config['scenario']['rating_pd_map'],
        score_to_rating_bounds=sample_config['scenario']['score_to_rating_bounds'],
        asset_correlation=sample_config['regulatory_params']['asset_correlation'],
        default_col='is_default',
    )
    
    # Verify results
    assert len(simulated_df) == 200, "Should return combined historical and application data, 150 historical + 50 application"
    assert 'simulated_pd' in simulated_df.columns
    assert 'simulated_rating' in simulated_df.columns
    assert (simulated_df['simulated_pd'] < 1.0).all(), "No loans should have PD=1.0"
