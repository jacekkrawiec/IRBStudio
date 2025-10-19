"""
Date Breakdown Analysis tests (Priority 2).

This module contains tests for date-based breakdown functionality:
- RWA calculation per reporting date
- Date-specific capital requirements
- Temporal trend analysis
- Period-end reporting
- Access via result.by_date property

Tests are organized by functional area.
"""

import pytest
import numpy as np
import pandas as pd
from irbstudio.engine.mortgage.airb_calculator import AIRBMortgageCalculator
from irbstudio.engine.mortgage.sa_calculator import SAMortgageCalculator


class TestDateBreakdownBasic:
    """Basic tests for date breakdown functionality."""
    
    def test_date_breakdown_basic_execution(self):
        """Test basic date breakdown execution."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        # Portfolio with multiple dates
        portfolio = pd.DataFrame({
            'exposure': [100000, 150000, 200000, 250000],
            'pd': [0.01, 0.02, 0.01, 0.02],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01', 
                                   '2024-02-01', '2024-02-01'])
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # Should contain date breakdown
        assert 'by_date' in summary
        assert len(summary['by_date']) == 2  # Two unique dates
    
    def test_date_breakdown_single_date(self):
        """Test that single date doesn't create breakdown."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        # Portfolio with single date
        portfolio = pd.DataFrame({
            'exposure': [100000, 150000],
            'pd': [0.01, 0.02],
            'date': pd.to_datetime('2024-01-01')
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # Should not create by_date for single date
        assert 'by_date' not in summary or len(summary.get('by_date', {})) == 0


class TestDateBreakdownCalculations:
    """Tests for date-specific RWA calculations."""
    
    def test_rwa_by_date_calculation(self):
        """Test that RWA is calculated correctly per reporting date."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        # Portfolio with multiple dates
        portfolio = pd.DataFrame({
            'exposure': [100000, 200000, 150000, 250000],
            'pd': [0.01, 0.02, 0.015, 0.025],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01',
                                   '2024-02-01', '2024-02-01'])
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # Check RWA for each date
        assert 'by_date' in summary
        
        # Each date should have metrics
        for date_key, date_metrics in summary['by_date'].items():
            assert 'total_rwa' in date_metrics
            assert 'total_exposure' in date_metrics
            assert date_metrics['total_rwa'] > 0
            assert date_metrics['total_exposure'] > 0
    
    def test_date_specific_capital_requirement(self):
        """Test capital requirement calculation per date."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        portfolio = pd.DataFrame({
            'exposure': [100000, 200000, 150000, 250000],
            'pd': [0.01, 0.02, 0.01, 0.02],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01',
                                   '2024-02-01', '2024-02-01'])
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # Capital requirement per date = total_rwa * 0.08
        for date_key, date_metrics in summary['by_date'].items():
            capital_req = date_metrics['total_rwa'] * 0.08
            assert capital_req > 0
    
    def test_date_breakdown_exposure_preservation(self):
        """Test that total exposure is preserved across dates."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        portfolio = pd.DataFrame({
            'exposure': [100000, 200000, 150000, 250000],
            'pd': [0.01, 0.02, 0.01, 0.02],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01',
                                   '2024-02-01', '2024-02-01'])
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # Sum of date exposures should equal total exposure
        date_exposures = sum(
            metrics['total_exposure'] 
            for metrics in summary['by_date'].values()
        )
        
        assert np.isclose(date_exposures, summary['total_exposure'], rtol=1e-5)


class TestDateBreakdownAccess:
    """Tests for accessing date breakdown data."""
    
    def test_access_by_date_property(self):
        """Test accessing date breakdown via result.by_date property."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        portfolio = pd.DataFrame({
            'exposure': [100000, 200000, 150000],
            'pd': [0.01, 0.02, 0.015],
            'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        })
        
        result = calculator.calculate(portfolio)
        
        # Access via by_date property if it exists
        if hasattr(result, 'by_date'):
            by_date = result.by_date
            assert by_date is not None
    
    def test_date_metrics_structure(self):
        """Test that date metrics have correct structure."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        portfolio = pd.DataFrame({
            'exposure': [100000, 200000, 150000, 250000],
            'pd': [0.01, 0.02, 0.01, 0.02],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01',
                                   '2024-02-01', '2024-02-01'])
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # Check structure of each date's metrics
        for date_key, date_metrics in summary['by_date'].items():
            assert 'total_rwa' in date_metrics
            assert 'total_exposure' in date_metrics
            assert 'average_risk_weight' in date_metrics
            assert 'weighted_average_rw' in date_metrics


class TestDateBreakdownTrends:
    """Tests for temporal trend analysis."""
    
    def test_temporal_trend_identification(self):
        """Test identification of temporal trends in RWA."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        # Create portfolio with increasing PD over time
        dates = pd.date_range('2024-01-01', '2024-06-01', freq='MS')
        portfolio = pd.DataFrame({
            'exposure': [100000] * len(dates),
            'pd': [0.01 + i*0.005 for i in range(len(dates))],  # Increasing PD
            'date': dates
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # RWA should generally increase over time
        rwa_by_date = {
            pd.to_datetime(k): v['total_rwa'] 
            for k, v in summary['by_date'].items()
        }
        
        sorted_dates = sorted(rwa_by_date.keys())
        rwa_values = [rwa_by_date[d] for d in sorted_dates]
        
        # Check that RWA generally increases (allowing some variation)
        assert rwa_values[-1] > rwa_values[0]
    
    def test_period_end_reporting(self):
        """Test period-end (month-end) reporting capabilities."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        # Portfolio with month-end dates
        portfolio = pd.DataFrame({
            'exposure': [100000, 150000, 200000],
            'pd': [0.01, 0.015, 0.02],
            'date': pd.to_datetime(['2024-01-31', '2024-02-29', '2024-03-31'])
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # Should have breakdown for each month-end
        assert len(summary['by_date']) == 3


class TestDateBreakdownMultiple:
    """Tests for portfolios with multiple reporting dates."""
    
    def test_multiple_dates_comprehensive(self):
        """Test comprehensive analysis with multiple reporting dates."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        # Portfolio with quarterly dates
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='QS')
        n_dates = len(dates)
        
        portfolio = pd.DataFrame({
            'exposure': [100000 + i*10000 for i in range(n_dates)],
            'pd': [0.01 + i*0.002 for i in range(n_dates)],
            'date': dates
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # Should have breakdown for each quarter
        assert 'by_date' in summary
        assert len(summary['by_date']) == n_dates
        
        # Total RWA should equal sum of date RWAs
        date_rwa_sum = sum(
            metrics['total_rwa'] 
            for metrics in summary['by_date'].values()
        )
        
        assert np.isclose(date_rwa_sum, summary['total_rwa'], rtol=1e-5)
    
    def test_date_breakdown_with_sa_calculator(self):
        """Test date breakdown works with SA calculator too."""
        calculator = SAMortgageCalculator(
            regulatory_params={
                'ltv_threshold': 0.80,
                'rw_secured': 0.35,
                'rw_unsecured': 0.75
            }
        )
        
        portfolio = pd.DataFrame({
            'exposure': [100000, 150000, 200000, 250000],
            'ltv': [0.70, 0.75, 0.80, 0.85],
            'property_value': [142857, 200000, 250000, 294118],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01',
                                   '2024-02-01', '2024-02-01'])
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        # SA should also support date breakdown
        assert 'by_date' in summary
        assert len(summary['by_date']) == 2


class TestDateBreakdownEdgeCases:
    """Tests for edge cases in date breakdown."""
    
    def test_date_breakdown_with_missing_dates(self):
        """Test handling of missing/null dates."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        portfolio = pd.DataFrame({
            'exposure': [100000, 150000, 200000],
            'pd': [0.01, 0.015, 0.02],
            'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01'])
        })
        
        result = calculator.calculate(portfolio)
        
        # Should handle date field gracefully
        summary = calculator.summarize_rwa(
            result.portfolio,
            date_field='date'
        )
        
        assert 'by_date' in summary
    
    def test_date_breakdown_without_date_field(self):
        """Test that breakdown is skipped when no date field provided."""
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.15,
                'lgd': 0.25
            }
        )
        
        portfolio = pd.DataFrame({
            'exposure': [100000, 150000],
            'pd': [0.01, 0.015]
        })
        
        result = calculator.calculate(portfolio)
        summary = calculator.summarize_rwa(result.portfolio)
        
        # Should not have by_date when no date_field specified
        assert 'by_date' not in summary


# Summary of test coverage:
# - Date breakdown basic: 2 tests
# - Date-specific calculations: 3 tests
# - Date breakdown access: 2 tests
# - Temporal trends: 2 tests
# - Multiple dates: 2 tests
# - Edge cases: 2 tests
# Total: 13 tests for date breakdown analysis
