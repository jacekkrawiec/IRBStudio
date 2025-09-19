import pandas as pd
import pytest
import numpy as np
from scipy.stats import norm

from irbstudio.engine.mortgage.airb_calculator import AIRBMortgageCalculator
from irbstudio.engine.mortgage.sa_calculator import SAMortgageCalculator
from irbstudio.engine.base import RWAResult

@pytest.fixture
def sample_portfolio():
    """Sample portfolio data for testing calculators."""
    return pd.DataFrame({
        'loan_id': ['L001', 'L002', 'L003', 'L004', 'L005'],
        'exposure': [100000, 200000, 150000, 300000, 250000],
        'pd': [0.01, 0.02, 0.05, 0.10, 0.15],
        'lgd': [0.25, 0.30, 0.20, 0.35, 0.40],
        'ltv': [0.45, 0.55, 0.75, 0.85, 1.05],
        'rating': ['1', '2', '3', '4', '5']
    })

@pytest.fixture
def airb_calculator():
    """Sample AIRB calculator for testing."""
    return AIRBMortgageCalculator({
        'asset_correlation': 0.15,
        'confidence_level': 0.999,
        'lgd': 0.25,
        'maturity_adjustment': False
    })

@pytest.fixture
def sa_calculator():
    """Sample SA calculator for testing."""
    return SAMortgageCalculator({})

def test_airb_calculator_initialization():
    """Test that the AIRB calculator initializes with correct parameters."""
    # Default parameters
    calc = AIRBMortgageCalculator({})
    assert calc.asset_correlation == 0.15
    assert calc.confidence_level == 0.999
    assert calc.lgd == 0.25
    assert not calc.maturity_adjustment
    
    # Custom parameters
    calc = AIRBMortgageCalculator({
        'asset_correlation': 0.10,
        'confidence_level': 0.995,
        'lgd': 0.30,
        'maturity_adjustment': True
    })
    assert calc.asset_correlation == 0.10
    assert calc.confidence_level == 0.995
    assert calc.lgd == 0.30
    assert calc.maturity_adjustment

def test_sa_calculator_initialization():
    """Test that the SA calculator initializes with correct parameters."""
    # Default parameters
    calc = SAMortgageCalculator({})
    assert len(calc.ltv_rw_map) == 6
    assert calc.ltv_rw_map[(0.0, 0.5)] == 0.20
    assert calc.ltv_rw_map[(1.0, float('inf'))] == 0.70
    
    # Custom parameters
    custom_map = {
        (0.0, 0.6): 0.15,
        (0.6, 1.0): 0.25,
        (1.0, float('inf')): 0.50
    }
    calc = SAMortgageCalculator({'ltv_rw_map': custom_map})
    assert len(calc.ltv_rw_map) == 3
    assert calc.ltv_rw_map[(0.0, 0.6)] == 0.15

def test_airb_risk_weight_calculation(airb_calculator, sample_portfolio):
    """Test AIRB risk weight calculation."""
    result = airb_calculator.calculate_rw(sample_portfolio)
    
    # Check that risk_weight column was added
    assert 'risk_weight' in result.columns
    
    # Check that all risk weights are positive
    assert (result['risk_weight'] > 0).all()
    
    # Manually calculate and verify the first row
    pd_val = sample_portfolio.iloc[0]['pd']
    lgd_val = sample_portfolio.iloc[0]['lgd']
    R = airb_calculator.asset_correlation
    
    # AIRB RW formula components for manual verification
    sqrt_1_minus_R = np.sqrt(1 - R)
    sqrt_R_div_1_minus_R = np.sqrt(R / (1 - R))
    
    # Normal inverse of PD and confidence level
    norm_inverse_pd = norm.ppf(pd_val)
    norm_inverse_conf = norm.ppf(airb_calculator.confidence_level)
    
    # Core AIRB formula
    N_term = norm.cdf(
        (norm_inverse_pd / sqrt_1_minus_R) + 
        (sqrt_R_div_1_minus_R * norm_inverse_conf)
    )
    
    # Calculate expected risk weight
    expected_rw = lgd_val * N_term * 12.5
    
    # Check that calculated risk weight matches expected (within tolerance)
    assert abs(result.iloc[0]['risk_weight'] - expected_rw) < 0.0001

def test_airb_rwa_calculation(airb_calculator, sample_portfolio):
    """Test AIRB RWA calculation."""
    result = airb_calculator.calculate_rwa(sample_portfolio)
    
    # Check that rwa column was added
    assert 'rwa' in result.columns
    
    # Check RWA calculation: RWA = exposure * risk_weight
    for i in range(len(result)):
        expected_rwa = result.iloc[i]['exposure'] * result.iloc[i]['risk_weight']
        assert abs(result.iloc[i]['rwa'] - expected_rwa) < 0.0001
    
    # Check summary statistics
    summary = airb_calculator.summarize_rwa(result)
    assert 'total_rwa' in summary
    assert 'average_risk_weight' in summary
    assert 'total_exposure' in summary
    
    # Verify total_exposure
    assert summary['total_exposure'] == sample_portfolio['exposure'].sum()

def test_sa_map_ltv_to_rw(sa_calculator):
    """Test LTV to risk weight mapping in SA calculator."""
    # Test each LTV range
    assert sa_calculator.map_ltv_to_rw(0.45) == 0.20  # LTV <= 50%
    assert sa_calculator.map_ltv_to_rw(0.55) == 0.25  # 50% < LTV <= 60%
    assert sa_calculator.map_ltv_to_rw(0.75) == 0.30  # 60% < LTV <= 80%
    assert sa_calculator.map_ltv_to_rw(0.85) == 0.40  # 80% < LTV <= 90%
    assert sa_calculator.map_ltv_to_rw(0.95) == 0.50  # 90% < LTV <= 100%
    assert sa_calculator.map_ltv_to_rw(1.05) == 0.70  # LTV > 100%

def test_sa_risk_weight_calculation(sa_calculator, sample_portfolio):
    """Test SA risk weight calculation."""
    result = sa_calculator.calculate_rw(sample_portfolio)
    
    # Check that risk_weight column was added
    assert 'risk_weight' in result.columns
    
    # Check risk weights for each row
    expected_rw = [0.20, 0.25, 0.30, 0.40, 0.70]
    for i in range(len(result)):
        assert abs(result.iloc[i]['risk_weight'] - expected_rw[i]) < 0.0001

def test_sa_rwa_calculation(sa_calculator, sample_portfolio):
    """Test SA RWA calculation."""
    result = sa_calculator.calculate_rwa(sample_portfolio)
    
    # Check that rwa column was added
    assert 'rwa' in result.columns
    
    # Check RWA calculation: RWA = exposure * risk_weight
    expected_rw = [0.20, 0.25, 0.30, 0.40, 0.70]
    for i in range(len(result)):
        expected_rwa = sample_portfolio.iloc[i]['exposure'] * expected_rw[i]
        assert abs(result.iloc[i]['rwa'] - expected_rwa) < 0.0001

def test_airb_calculate_method(airb_calculator, sample_portfolio):
    """Test the calculate method of AIRB calculator."""
    result = airb_calculator.calculate(sample_portfolio)
    
    # Check return type
    assert isinstance(result, RWAResult)
    
    # Check properties
    assert result.total_exposure == sample_portfolio['exposure'].sum()
    assert result.capital_requirement == result.total_rwa * 0.08
    
    # Check metadata
    assert result.metadata['calculator_type'] == 'AIRB'
    assert result.metadata['asset_class'] == 'Mortgage'
    assert 'average_pd' in result.metadata
    
    # Check string representation
    str_repr = str(result)
    assert "RWA Calculation Result:" in str_repr
    assert "Total Exposure:" in str_repr
    assert "Total RWA:" in str_repr
    assert "Capital Requirement:" in str_repr

def test_sa_calculate_method(sa_calculator, sample_portfolio):
    """Test the calculate method of SA calculator."""
    result = sa_calculator.calculate(sample_portfolio)
    
    # Check return type
    assert isinstance(result, RWAResult)
    
    # Check properties
    assert result.total_exposure == sample_portfolio['exposure'].sum()
    assert result.capital_requirement == result.total_rwa * 0.08
    
    # Check metadata
    assert result.metadata['calculator_type'] == 'SA'
    assert result.metadata['asset_class'] == 'Mortgage'
    assert 'average_ltv' in result.metadata
    
    # Check string representation
    str_repr = str(result)
    assert "RWA Calculation Result:" in str_repr
    assert "Total Exposure:" in str_repr
    assert "Total RWA:" in str_repr
    assert "Capital Requirement:" in str_repr

def test_sa_vs_airb_comparison(sa_calculator, airb_calculator, sample_portfolio):
    """Compare SA and AIRB results for the same portfolio."""
    sa_result = sa_calculator.calculate(sample_portfolio)
    airb_result = airb_calculator.calculate(sample_portfolio)
    
    # Both should have the same total exposure
    assert sa_result.total_exposure == airb_result.total_exposure
    
    # AIRB is typically more risk-sensitive, but this will depend on the specific portfolio
    # This is more of a sanity check than a strict requirement
    print(f"SA Total RWA: {sa_result.total_rwa}")
    print(f"AIRB Total RWA: {airb_result.total_rwa}")
    
    # Check that both calculated something reasonable
    assert sa_result.total_rwa > 0
    assert airb_result.total_rwa > 0

def test_validate_inputs_missing_columns(airb_calculator, sa_calculator):
    """Test validation of required columns."""
    # Missing 'exposure'
    invalid_df = pd.DataFrame({'pd': [0.01, 0.02]})
    with pytest.raises(ValueError):
        airb_calculator.validate_inputs(invalid_df, airb_calculator.required_columns)
    
    # Missing 'ltv'
    invalid_df = pd.DataFrame({'exposure': [100000, 200000]})
    with pytest.raises(ValueError):
        sa_calculator.validate_inputs(invalid_df, sa_calculator.required_columns)

def test_summarize_rwa_missing_rwa_column(airb_calculator):
    """Test error handling when summarizing without RWA column."""
    df = pd.DataFrame({'exposure': [100000], 'risk_weight': [0.2]})
    with pytest.raises(ValueError):
        airb_calculator.summarize_rwa(df)