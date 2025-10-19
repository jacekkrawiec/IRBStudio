"""
Edge case tests for IRBStudio.

Priority 3: Edge Cases (focusing on most critical edge cases)
- Data edge cases
- Configuration edge cases  
- Calculation edge cases
"""

import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError

from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.engine.mortgage import AIRBMortgageCalculator, SAMortgageCalculator
from irbstudio.config.schema import Scenario, ColumnMapping


class TestDataEdgeCases:
    """Test edge cases in portfolio data."""
    
    def test_edge_case_single_loan(self, score_to_rating_bounds):
        """Test portfolio with single loan."""
        # Create minimal portfolio with one loan and multiple dates for Beta Mixture
        dates = pd.date_range('2023-07-01', '2024-12-31', freq='ME')
        
        # Create rows for same loan across multiple dates
        rows = []
        for date in dates:
            rows.append({
                'loan_id': 'SINGLE_LOAN',
                'exposure': 100000,
                'rating': 'A',
                'score': 0.05,
                'reporting_date': date,
                'default_flag': 0,
                'into_default_flag': 0
            })
        
        single_loan_df = pd.DataFrame(rows)
        single_loan_df['reporting_date'] = pd.to_datetime(single_loan_df['reporting_date'])
        
        # Should handle single loan
        simulator = PortfolioSimulator(
            portfolio_df=single_loan_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score'
        )
        
        result = simulator.simulate_once(random_seed=42)
        assert result is not None
        assert len(result) >= 1
    
    def test_edge_case_all_defaults(self, small_portfolio_df):
        """Test portfolio where all loans have defaulted."""
        # Mark all loans as defaulted
        df_all_defaults = small_portfolio_df.copy()
        df_all_defaults['default_flag'] = 1
        
        # Calculator should handle this
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        
        result = calculator.calculate(df_all_defaults)
        assert result is not None
        assert 'total_rwa' in result.summary
    
    def test_edge_case_no_defaults(self, small_portfolio_df):
        """Test portfolio with no defaults."""
        # Mark no loans as defaulted
        df_no_defaults = small_portfolio_df.copy()
        df_no_defaults['default_flag'] = 0
        
        # Calculator should handle this
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        
        result = calculator.calculate(df_no_defaults)
        assert result is not None
        assert 'total_rwa' in result.summary
        assert result.summary['total_rwa'] > 0
    
    def test_edge_case_extreme_pd_values(self, small_portfolio_df):
        """Test with extreme PD values (0 and 1)."""
        df_extreme = small_portfolio_df.copy()
        
        # Mix of extreme PD values
        n = len(df_extreme)
        df_extreme.loc[:n//2, 'pd'] = 0.001  # Very low PD
        df_extreme.loc[n//2:, 'pd'] = 0.999  # Very high PD
        
        # Calculator should handle extreme values
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        
        result = calculator.calculate(df_extreme)
        assert result is not None
        assert 'total_rwa' in result.summary
    
    def test_edge_case_zero_exposures(self, small_portfolio_df):
        """Test with some zero exposures."""
        df_zero = small_portfolio_df.copy()
        
        # Set some exposures to zero
        df_zero.loc[:10, 'exposure'] = 0
        
        # Calculator should handle zero exposures
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        
        result = calculator.calculate(df_zero)
        assert result is not None
        assert 'total_rwa' in result.summary
        
        # Total exposure should only count non-zero loans
        assert result.summary['total_exposure'] > 0
    
    def test_edge_case_duplicate_loan_ids(self, score_to_rating_bounds):
        """Test handling of duplicate loan IDs (across dates is valid)."""
        # Create portfolio with same loan_id across multiple dates
        dates = pd.date_range('2023-07-01', '2024-12-31', freq='ME')
        
        rows = []
        for date in dates:
            rows.append({
                'loan_id': 'LOAN_001',  # Same ID
                'exposure': 100000,
                'rating': 'A',
                'score': 0.05,
                'reporting_date': date,
                'default_flag': 0,
                'into_default_flag': 0
            })
        
        df_dups = pd.DataFrame(rows)
        df_dups['reporting_date'] = pd.to_datetime(df_dups['reporting_date'])
        
        # Should handle same loan across dates
        simulator = PortfolioSimulator(
            portfolio_df=df_dups,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score'
        )
        
        result = simulator.simulate_once(random_seed=42)
        assert result is not None


class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""
    
    def test_edge_case_target_auc_minimum(self):
        """Test minimum valid AUC (0.5 = random classifier)."""
        # AUC must be > 0.5 according to schema
        with pytest.raises(ValidationError):
            Scenario(
                name='test',
                pd_auc=0.5,  # At boundary, should fail (gt=0.5)
                portfolio_default_rate=0.03,
                lgd=0.25
            )
    
    def test_edge_case_target_auc_near_minimum(self):
        """Test AUC just above minimum."""
        scenario = Scenario(
            name='test',
            pd_auc=0.51,  # Just above minimum
            portfolio_default_rate=0.03,
            lgd=0.25
        )
        assert scenario.pd_auc == 0.51
    
    def test_edge_case_target_auc_near_maximum(self):
        """Test AUC near maximum (< 1.0)."""
        scenario = Scenario(
            name='test',
            pd_auc=0.99,  # Near maximum
            portfolio_default_rate=0.03,
            lgd=0.25
        )
        assert scenario.pd_auc == 0.99
    
    def test_edge_case_target_auc_maximum(self):
        """Test maximum AUC boundary (1.0 = perfect classifier)."""
        # AUC must be < 1.0 according to schema
        with pytest.raises(ValidationError):
            Scenario(
                name='test',
                pd_auc=1.0,  # At boundary, should fail (lt=1.0)
                portfolio_default_rate=0.03,
                lgd=0.25
            )
    
    def test_edge_case_zero_new_loans(self):
        """Test scenario with zero new loans."""
        scenario = Scenario(
            name='test',
            pd_auc=0.75,
            portfolio_default_rate=0.03,
            lgd=0.25,
            new_loan_rate=0.0  # No new loans
        )
        assert scenario.new_loan_rate == 0.0
    
    def test_edge_case_asset_correlation_zero(self):
        """Test zero asset correlation."""
        # Zero correlation is valid
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'asset_correlation': 0.0,
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        assert calculator is not None
    
    def test_edge_case_single_iteration(self, small_portfolio_df, score_to_rating_bounds):
        """Test with single Monte Carlo iteration."""
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score'
        )
        
        # Single iteration should work
        results = simulator.run_monte_carlo(num_iterations=1, random_seed=42)
        
        assert results is not None
        assert len(results) == 1


class TestCalculationEdgeCases:
    """Test edge cases in RWA calculations."""
    
    def test_edge_case_very_small_portfolio(self, score_to_rating_bounds):
        """Test calculation with very small portfolio."""
        # Create minimal 3-loan portfolio with date range
        dates = pd.date_range('2023-07-01', '2024-12-31', freq='ME')
        
        rows = []
        for loan_id in ['L1', 'L2', 'L3']:
            for date in dates:
                rows.append({
                    'loan_id': loan_id,
                    'exposure': 50000,
                    'pd': 0.05,
                    'rating': 'BBB',
                    'score': 0.05,
                    'reporting_date': date,
                    'default_flag': 0,
                    'into_default_flag': 0
                })
        
        tiny_df = pd.DataFrame(rows)
        tiny_df['reporting_date'] = pd.to_datetime(tiny_df['reporting_date'])
        
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        
        result = calculator.calculate(tiny_df)
        assert result is not None
        assert 'total_rwa' in result.summary
        assert result.summary['total_rwa'] > 0
    
    def test_edge_case_uniform_portfolio(self, score_to_rating_bounds):
        """Test portfolio where all loans are identical."""
        # Create uniform portfolio with date range
        dates = pd.date_range('2023-07-01', '2024-12-31', freq='ME')
        
        rows = []
        for i, date in enumerate(dates):
            rows.append({
                'loan_id': f'L{i}',
                'exposure': 100000,  # All same
                'pd': 0.05,  # All same
                'rating': 'BBB',  # All same
                'score': 0.05,  # All same
                'reporting_date': date,
                'default_flag': 0,
                'into_default_flag': 0
            })
        
        uniform_df = pd.DataFrame(rows)
        uniform_df['reporting_date'] = pd.to_datetime(uniform_df['reporting_date'])
        
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        
        result = calculator.calculate(uniform_df)
        assert result is not None
        assert 'total_rwa' in result.summary
    
    def test_edge_case_missing_optional_columns(self, small_portfolio_df):
        """Test calculation with minimal required columns only."""
        # Keep only required columns
        minimal_df = small_portfolio_df[['loan_id', 'exposure']].copy()
        
        # Add pd for calculation
        minimal_df['pd'] = 0.05
        
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        
        result = calculator.calculate(minimal_df)
        assert result is not None
        assert 'total_rwa' in result.summary
