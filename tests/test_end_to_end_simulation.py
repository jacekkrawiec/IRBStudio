import pandas as pd
import pytest
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import DateOffset

from irbstudio.simulation.pd_simulator import simulate_portfolio
from irbstudio.engine.mortgage.airb_calculator import AIRBMortgageCalculator
from irbstudio.engine.mortgage.sa_calculator import SAMortgageCalculator

@pytest.fixture
def sample_historical_data():
    """Generate a realistic historical dataset for testing the end-to-end flow."""
    # Generate a 3-year dataset with 100 loans
    n_loans = 100
    loan_ids = [f'L{i:03d}' for i in range(n_loans)]
    
    # Generate dates from 2020 to 2023
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 1, 1)
    application_date = datetime(2022, 7, 1)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='3M')
    
    # Create all combinations of loan_ids and dates
    data = []
    for loan_id in loan_ids:
        for date in dates:
            # Skip combinations that would create an application entry before historical
            if date < application_date or loan_id < 'L080':  # Existing loans
                data.append({
                    'loan_id': loan_id,
                    'date': date,
                    'rating': np.random.choice(['1', '2', '3', '4', '5'], p=[0.1, 0.2, 0.4, 0.2, 0.1]),
                    'pd': np.random.beta(2, 8),  # Generate PDs between 0 and 1 with most < 0.3
                    'is_default': 0,
                    'exposure': np.random.uniform(50000, 500000),
                    'ltv': np.random.uniform(0.4, 1.1)
                })
    
    df = pd.DataFrame(data)
    
    # Add some defaults (5% default rate)
    default_loan_ids = np.random.choice(loan_ids[:80], size=int(n_loans * 0.05), replace=False)
    
    for loan_id in default_loan_ids:
        # Choose a random date for default after the first year
        default_date_idx = np.random.randint(4, len(dates))
        default_date = dates[default_date_idx]
        
        # Set the default flag for this date
        default_idx = df[(df['loan_id'] == loan_id) & (df['date'] == default_date)].index
        if len(default_idx) > 0:
            df.loc[default_idx[0], 'is_default'] = 1
            
            # Set all subsequent dates to rating 'D' and PD 1.0
            for date in dates[default_date_idx:]:
                idx = df[(df['loan_id'] == loan_id) & (df['date'] == date)].index
                if len(idx) > 0:
                    df.loc[idx[0], 'rating'] = 'D'
                    df.loc[idx[0], 'pd'] = 1.0
    
    return df

@pytest.fixture
def score_to_rating_bounds():
    """Sample mapping of scores to rating bounds."""
    return {
        '1': (0, 0.05),
        '2': (0.05, 0.10),
        '3': (0.10, 0.20),
        '4': (0.20, 0.30),
        '5': (0.30, 1.0),
        # Special case for default, needs to be handled separately in the code
        'D': (0, 0)  # Default rating (not used in normal mapping)
    }

def test_end_to_end_simulation_with_rwa_calculation(sample_historical_data, score_to_rating_bounds):
    """Test the end-to-end flow from portfolio simulation to RWA calculation."""
    # Step 1: Run portfolio simulation
    application_start_date = datetime(2022, 7, 1)
    
    simulated_portfolio = simulate_portfolio(
        portfolio_df=sample_historical_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='is_default',
        score_col='pd',
        application_start_date=application_start_date,
        target_auc=0.75
    )
    
    # Verify simulation results
    assert 'simulated_rating' in simulated_portfolio.columns
    assert 'simulated_pd' in simulated_portfolio.columns
    
    # Step 2: Filter to get just the application portfolio (most recent date)
    most_recent_date = simulated_portfolio['date'].max()
    application_portfolio = simulated_portfolio[simulated_portfolio['date'] == most_recent_date].copy()
    
    # Ensure required columns are present
    application_portfolio.rename(columns={'simulated_pd': 'pd'}, inplace=True)
    
    # Add property_value column for SA calculator
    application_portfolio['property_value'] = application_portfolio['exposure'] / application_portfolio['ltv']
    
    # Step 3: Calculate RWA using AIRB calculator
    airb_calculator = AIRBMortgageCalculator({
        'asset_correlation': 0.15,
        'confidence_level': 0.999,
        'lgd': 0.25
    })
    
    airb_result = airb_calculator.calculate(application_portfolio)
    
    # Verify AIRB results
    assert airb_result.total_rwa > 0
    assert airb_result.capital_requirement == airb_result.total_rwa * 0.08
    assert airb_result.metadata['calculator_type'] == 'AIRB'
    
    # Step 4: Calculate RWA using SA calculator
    sa_calculator = SAMortgageCalculator({})
    
    sa_result = sa_calculator.calculate(application_portfolio)
    
    # Verify SA results
    assert sa_result.total_rwa > 0
    assert sa_result.capital_requirement == sa_result.total_rwa * 0.08
    assert sa_result.metadata['calculator_type'] == 'SA'
    
    # Step 5: Compare AIRB and SA results
    print(f"AIRB Total RWA: {airb_result.total_rwa:,.2f}")
    print(f"SA Total RWA: {sa_result.total_rwa:,.2f}")
    print(f"Difference: {airb_result.total_rwa - sa_result.total_rwa:,.2f}")
    print(f"Percentage Difference: {(airb_result.total_rwa / sa_result.total_rwa - 1) * 100:.2f}%")
    
    # There should be a difference between SA and AIRB
    assert airb_result.total_rwa != sa_result.total_rwa
    
    # Both should have calculated valid results
    assert airb_result.total_exposure == sa_result.total_exposure
    assert airb_result.total_rwa > 0
    assert sa_result.total_rwa > 0
    
    # Verify portfolio attributes in results
    assert len(airb_result.portfolio) == len(application_portfolio)
    assert 'risk_weight' in airb_result.portfolio.columns
    assert 'rwa' in airb_result.portfolio.columns