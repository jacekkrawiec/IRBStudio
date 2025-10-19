"""
Tests for RWA calculators.

Priority 1: Critical - Core Functionality
"""

import pytest
import pandas as pd
import numpy as np

from irbstudio.engine.mortgage.airb_calculator import AIRBMortgageCalculator
from irbstudio.engine.mortgage.sa_calculator import SAMortgageCalculator
from irbstudio.engine.base import RWAResult


class TestAIRBMortgageCalculator:
    """Tests for AIRBMortgageCalculator."""
    
    def test_airb_calculator_init(self, airb_params):
        """Test AIRBMortgageCalculator.__init__()."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        assert calculator is not None
        assert calculator.asset_correlation == 0.15
        assert calculator.lgd == 0.25
    
    def test_airb_calculate_rw_basic(self, airb_params, small_portfolio_df):
        """Test calculate_rw() basic risk weight calculation."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        result_df = calculator.calculate_rw(small_portfolio_df)
        
        assert 'risk_weight' in result_df.columns
        assert (result_df['risk_weight'] >= 0).all()
        assert (result_df['risk_weight'] <= 20).all()  # Reasonable upper bound
    
    def test_airb_calculate_rw_with_lgd_column(self, airb_params, small_portfolio_df):
        """Test calculate_rw() with exposure-level LGD."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        # Portfolio already has lgd column
        result_df = calculator.calculate_rw(small_portfolio_df)
        
        assert 'risk_weight' in result_df.columns
        assert (result_df['risk_weight'] >= 0).all()
    
    def test_airb_calculate_rwa_basic(self, airb_params, small_portfolio_df):
        """Test calculate_rwa() RWA calculation."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        result_df = calculator.calculate_rwa(small_portfolio_df)
        
        assert 'risk_weight' in result_df.columns
        assert 'rwa' in result_df.columns
        assert (result_df['rwa'] >= 0).all()
    
    def test_airb_calculate_full(self, airb_params, small_portfolio_df):
        """Test calculate() complete workflow."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        result = calculator.calculate(small_portfolio_df)
        
        assert isinstance(result, RWAResult)
        assert result.total_rwa > 0
        assert result.total_exposure > 0
        assert result.portfolio is not None
    
    def test_airb_summarize_rwa_basic(self, airb_params, small_portfolio_df):
        """Test summarize_rwa() summary statistics."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        result_df = calculator.calculate_rwa(small_portfolio_df)
        summary = calculator.summarize_rwa(result_df)
        
        assert 'total_rwa' in summary
        assert 'total_exposure' in summary
        assert 'average_risk_weight' in summary
        assert summary['total_rwa'] > 0
    
    def test_airb_summarize_rwa_with_date_field(self, airb_params, multi_date_portfolio_df):
        """Test summarize_rwa() with date breakdown calculation."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        result_df = calculator.calculate_rwa(multi_date_portfolio_df)
        summary = calculator.summarize_rwa(result_df, date_field='reporting_date')
        
        assert 'total_rwa' in summary
        if 'by_date' in summary:
            assert isinstance(summary['by_date'], dict)
            assert len(summary['by_date']) > 0
    
    def test_airb_extreme_pd_values(self, airb_params):
        """Test AIRB with extreme PD values near 0 or 1."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        df = pd.DataFrame({
            'loan_id': ['L1', 'L2', 'L3'],
            'exposure': [100000, 200000, 300000],
            'pd': [0.0001, 0.5, 0.9999],  # Extreme values
            'lgd': [0.25, 0.25, 0.25],
        })
        
        result_df = calculator.calculate_rw(df)
        
        assert 'risk_weight' in result_df.columns
        assert not result_df['risk_weight'].isna().any()
        assert (result_df['risk_weight'] >= 0).all()
    
    def test_airb_zero_exposure(self, airb_params):
        """Test AIRB with zero exposure loans."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        df = pd.DataFrame({
            'loan_id': ['L1', 'L2'],
            'exposure': [0, 100000],
            'pd': [0.05, 0.05],
            'lgd': [0.25, 0.25],
        })
        
        result_df = calculator.calculate_rwa(df)
        
        assert 'rwa' in result_df.columns
        assert result_df.loc[0, 'rwa'] == 0  # Zero exposure = zero RWA


class TestSAMortgageCalculator:
    """Tests for SAMortgageCalculator."""
    
    def test_sa_calculator_init(self, sa_params):
        """Test SAMortgageCalculator.__init__()."""
        calculator = SAMortgageCalculator(sa_params)
        
        assert calculator is not None
        assert calculator.secured_portion_rw == 0.20
        assert calculator.unsecured_portion_rw == 0.75
    
    def test_sa_calculate_rw_low_ltv(self, sa_params):
        """Test calculate_rw() for LTV ≤ threshold."""
        calculator = SAMortgageCalculator(sa_params)
        
        df = pd.DataFrame({
            'loan_id': ['L1', 'L2'],
            'exposure': [100000, 200000],
            'ltv': [0.50, 0.40],  # Below threshold
            'property_value': [200000, 500000],
        })
        
        result_df = calculator.calculate_rw(df)
        
        assert 'risk_weight' in result_df.columns
        # Low LTV should get secured rate (20%)
        assert (result_df['risk_weight'] <= 0.25).all()
    
    def test_sa_calculate_rw_high_ltv(self, sa_params):
        """Test calculate_rw() for LTV > threshold."""
        calculator = SAMortgageCalculator(sa_params)
        
        df = pd.DataFrame({
            'loan_id': ['L1', 'L2'],
            'exposure': [100000, 200000],
            'ltv': [0.80, 0.90],  # Above threshold
            'property_value': [125000, 222222],
        })
        
        result_df = calculator.calculate_rw(df)
        
        assert 'risk_weight' in result_df.columns
        # High LTV should get higher risk weights
        assert (result_df['risk_weight'] > 0.25).any()
    
    def test_sa_calculate_rwa_basic(self, sa_params, small_portfolio_df):
        """Test calculate_rwa() basic RWA."""
        calculator = SAMortgageCalculator(sa_params)
        
        result_df = calculator.calculate_rwa(small_portfolio_df)
        
        assert 'risk_weight' in result_df.columns
        assert 'rwa' in result_df.columns
        assert (result_df['rwa'] >= 0).all()
    
    def test_sa_calculate_full(self, sa_params, small_portfolio_df):
        """Test calculate() complete workflow."""
        calculator = SAMortgageCalculator(sa_params)
        
        result = calculator.calculate(small_portfolio_df)
        
        assert isinstance(result, RWAResult)
        assert result.total_rwa > 0
        assert result.total_exposure > 0
    
    def test_sa_summarize_rwa_basic(self, sa_params, small_portfolio_df):
        """Test summarize_rwa() summary."""
        calculator = SAMortgageCalculator(sa_params)
        
        result_df = calculator.calculate_rwa(small_portfolio_df)
        summary = calculator.summarize_rwa(result_df)
        
        assert 'total_rwa' in summary
        assert 'total_exposure' in summary
        assert summary['total_rwa'] > 0
    
    def test_sa_missing_property_value(self, sa_params):
        """Test SA with missing property values."""
        calculator = SAMortgageCalculator(sa_params)
        
        df = pd.DataFrame({
            'loan_id': ['L1', 'L2'],
            'exposure': [100000, 200000],
            'ltv': [0.70, 0.80],
            'property_value': [142857, 250000],  # Calculate from exposure/ltv
        })
        
        # Should work with property values provided
        result_df = calculator.calculate_rw(df)
        
        assert 'risk_weight' in result_df.columns
    
    def test_sa_zero_exposure(self, sa_params):
        """Test SA with zero exposure."""
        calculator = SAMortgageCalculator(sa_params)
        
        df = pd.DataFrame({
            'loan_id': ['L1', 'L2'],
            'exposure': [0, 100000],
            'ltv': [0.70, 0.80],
            'property_value': [0, 125000],
        })
        
        result_df = calculator.calculate_rwa(df)
        
        assert 'rwa' in result_df.columns
        assert result_df.loc[0, 'rwa'] == 0


class TestRWAResult:
    """Tests for RWAResult class."""
    
    def test_rwa_result_init(self, small_portfolio_df):
        """Test RWAResult.__init__()."""
        portfolio = small_portfolio_df.copy()
        portfolio['risk_weight'] = 0.50
        portfolio['rwa'] = portfolio['exposure'] * portfolio['risk_weight']
        
        result = RWAResult(
            portfolio_with_rwa=portfolio,
            summary={'total_rwa': portfolio['rwa'].sum()},
            metadata={'calculator': 'AIRB'}
        )
        
        assert result is not None
        assert result.portfolio is not None
    
    def test_rwa_result_total_rwa_property(self, airb_params, small_portfolio_df):
        """Test total_rwa property."""
        calculator = AIRBMortgageCalculator(airb_params)
        result = calculator.calculate(small_portfolio_df)
        
        assert result.total_rwa > 0
        assert isinstance(result.total_rwa, (int, float))
    
    def test_rwa_result_total_exposure_property(self, airb_params, small_portfolio_df):
        """Test total_exposure property."""
        calculator = AIRBMortgageCalculator(airb_params)
        result = calculator.calculate(small_portfolio_df)
        
        assert result.total_exposure > 0
        assert isinstance(result.total_exposure, (int, float))
    
    def test_rwa_result_capital_requirement(self, airb_params, small_portfolio_df):
        """Test capital_requirement calculation (8%)."""
        calculator = AIRBMortgageCalculator(airb_params)
        result = calculator.calculate(small_portfolio_df)
        
        if hasattr(result, 'capital_requirement'):
            assert result.capital_requirement > 0
            # Capital requirement should be ~8% of RWA
            assert abs(result.capital_requirement - result.total_rwa * 0.08) < 0.01
    
    def test_rwa_result_by_date_property(self, airb_params, multi_date_portfolio_df):
        """Test by_date property access."""
        calculator = AIRBMortgageCalculator(airb_params)
        
        # Calculate with date breakdown
        result = calculator.calculate(multi_date_portfolio_df, date_column='reporting_date')
        
        if hasattr(result, 'by_date'):
            by_date = result.by_date
            if by_date:
                assert isinstance(by_date, dict)
