import pandas as pd
import pytest
import numpy as np

from irbstudio.engine.mortgage.sa_calculator import SAMortgageCalculator

@pytest.fixture
def sample_mortgage_portfolio():
    """Create a sample mortgage portfolio for testing."""
    # Create 100 sample loans with different exposures and property values
    n_loans = 100
    np.random.seed(42)  # For reproducible results
    
    data = []
    for i in range(n_loans):
        # Create loans with different characteristics
        exposure = np.random.uniform(50000, 500000)
        property_value = exposure / np.random.uniform(0.5, 1.2)  # Various LTVs
        
        data.append({
            'loan_id': f'L{i:03d}',
            'exposure': exposure,
            'property_value': property_value,
            'ltv': exposure / property_value
        })
    
    return pd.DataFrame(data)

def test_sa_calculator_initialization():
    """Test that the SAMortgageCalculator initializes with correct default values."""
    # Initialize with default parameters
    calculator = SAMortgageCalculator({})
    
    assert calculator.secured_portion_rw == 0.20
    assert calculator.unsecured_portion_rw == 0.75
    assert calculator.property_value_threshold == 0.55
    assert set(calculator.required_columns) == {'exposure', 'property_value'}
    
    # Initialize with custom parameters
    custom_calculator = SAMortgageCalculator({
        'secured_portion_rw': 0.15,
        'unsecured_portion_rw': 0.80,
        'property_value_threshold': 0.60
    })
    
    assert custom_calculator.secured_portion_rw == 0.15
    assert custom_calculator.unsecured_portion_rw == 0.80
    assert custom_calculator.property_value_threshold == 0.60

def test_loan_splitting_approach(sample_mortgage_portfolio):
    """Test the CRR3 loan splitting approach implementation."""
    calculator = SAMortgageCalculator({})
    result_df = calculator.calculate_rw(sample_mortgage_portfolio)
    
    # Check that all required columns are added
    expected_columns = [
        'secured_threshold', 'secured_portion', 'unsecured_portion',
        'secured_rwa', 'unsecured_rwa', 'rwa', 'risk_weight'
    ]
    for col in expected_columns:
        assert col in result_df.columns
    
    # Validate calculations for each loan
    for _, row in result_df.iterrows():
        # Check that secured threshold is calculated correctly
        assert row['secured_threshold'] == row['property_value'] * calculator.property_value_threshold
        
        # Check that secured portion is correctly capped
        assert row['secured_portion'] == min(row['exposure'], row['secured_threshold'])
        
        # Check that unsecured portion is the remainder
        assert np.isclose(row['unsecured_portion'], row['exposure'] - row['secured_portion'])
        
        # Check RWA calculations
        assert np.isclose(row['secured_rwa'], row['secured_portion'] * calculator.secured_portion_rw)
        assert np.isclose(row['unsecured_rwa'], row['unsecured_portion'] * calculator.unsecured_portion_rw)
        assert np.isclose(row['rwa'], row['secured_rwa'] + row['unsecured_rwa'])
        
        # Check effective risk weight
        if row['exposure'] > 0:
            assert np.isclose(row['risk_weight'], row['rwa'] / row['exposure'])
        else:
            assert row['risk_weight'] == 0

def test_end_to_end_calculation(sample_mortgage_portfolio):
    """Test the end-to-end calculation flow."""
    calculator = SAMortgageCalculator({})
    result = calculator.calculate(sample_mortgage_portfolio)
    
    # Check that the result has the correct structure
    assert hasattr(result, 'portfolio')
    assert hasattr(result, 'summary')
    assert hasattr(result, 'metadata')
    
    # Check portfolio results
    assert len(result.portfolio) == len(sample_mortgage_portfolio)
    assert 'rwa' in result.portfolio.columns
    assert 'risk_weight' in result.portfolio.columns
    
    # Check summary stats
    assert 'total_exposure' in result.summary
    assert 'total_rwa' in result.summary
    assert 'average_risk_weight' in result.summary
    
    # Add the capital requirement calculation in our test
    capital_requirement = result.summary['total_rwa'] * 0.08
    
    # Check metadata
    assert result.metadata['calculator_type'] == 'SA'
    assert result.metadata['regulatory_framework'] == 'CRR3'
    assert 'avg_secured_portion_pct' in result.metadata
    assert 'avg_unsecured_portion_pct' in result.metadata
    
    # Capital requirement should be 8% of total RWA
    capital_requirement = result.summary['total_rwa'] * 0.08
    assert capital_requirement > 0

def test_special_cases():
    """Test special cases and edge conditions."""
    # Create a portfolio with edge cases
    edge_cases = pd.DataFrame([
        # Zero exposure
        {'loan_id': 'E001', 'exposure': 0, 'property_value': 100000},
        
        # Zero property value
        {'loan_id': 'E002', 'exposure': 100000, 'property_value': 0},
        
        # Exposure exactly equal to secured threshold
        {'loan_id': 'E003', 'exposure': 55000, 'property_value': 100000},
        
        # Very high LTV (> 100%)
        {'loan_id': 'E004', 'exposure': 200000, 'property_value': 150000},
        
        # Very low LTV (< 20%)
        {'loan_id': 'E005', 'exposure': 10000, 'property_value': 100000}
    ])
    
    calculator = SAMortgageCalculator({})
    result_df = calculator.calculate_rw(edge_cases)
    
    # Zero exposure should have zero risk weight and zero RWA
    zero_exp_row = result_df[result_df['loan_id'] == 'E001'].iloc[0]
    assert zero_exp_row['risk_weight'] == 0
    assert zero_exp_row['rwa'] == 0
    
    # Zero property value should have 100% unsecured portion
    zero_prop_row = result_df[result_df['loan_id'] == 'E002'].iloc[0]
    assert zero_prop_row['secured_threshold'] == 0
    assert zero_prop_row['secured_portion'] == 0
    assert zero_prop_row['unsecured_portion'] == 100000
    assert zero_prop_row['risk_weight'] == calculator.unsecured_portion_rw
    
    # Exposure exactly equal to secured threshold
    equal_row = result_df[result_df['loan_id'] == 'E003'].iloc[0]
    assert np.isclose(equal_row['secured_threshold'], 55000)
    assert np.isclose(equal_row['secured_portion'], 55000)
    assert np.isclose(equal_row['unsecured_portion'], 0)
    assert np.isclose(equal_row['risk_weight'], calculator.secured_portion_rw)
    
    # Very high LTV
    high_ltv_row = result_df[result_df['loan_id'] == 'E004'].iloc[0]
    assert high_ltv_row['secured_threshold'] == 150000 * 0.55
    assert high_ltv_row['secured_portion'] == 150000 * 0.55
    assert high_ltv_row['unsecured_portion'] == 200000 - (150000 * 0.55)
    
    # Very low LTV
    low_ltv_row = result_df[result_df['loan_id'] == 'E005'].iloc[0]
    assert low_ltv_row['secured_portion'] == 10000
    assert low_ltv_row['unsecured_portion'] == 0
    assert low_ltv_row['risk_weight'] == calculator.secured_portion_rw