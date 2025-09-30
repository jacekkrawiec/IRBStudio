"""
Tests for the enhanced RWA breakdown capabilities.
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any

from irbstudio.engine.base import BaseRWACalculator, RWAResult


# Test implementation of the calculator
class SimpleTestCalculator(BaseRWACalculator):
    """Simple calculator implementation for testing."""

    def calculate_rw(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk weights."""
        df = portfolio_df.copy()
        df['risk_weight'] = df['pd'].apply(lambda x: max(0.5, min(1.5, x * 10)))
        return df

    def calculate_rwa(self, 
                     portfolio_df: pd.DataFrame,
                     date_column: str = None) -> pd.DataFrame:
        """Calculate RWA."""
        df = self.calculate_rw(portfolio_df)
        df['rwa'] = df['exposure'] * df['risk_weight']
        return df

    def calculate(self, portfolio_df: pd.DataFrame) -> RWAResult:
        """Calculate RWA and return a RWAResult object."""
        required_cols = ['pd', 'exposure']
        self.validate_inputs(portfolio_df, required_cols)
        
        with_rwa = self.calculate_rwa(portfolio_df)
        
        # Use new parametrized summary method
        summary = self.summarize_rwa(
            with_rwa,
            breakdown_fields=['rating', 'segment'],
            date_field='reporting_date' if 'reporting_date' in portfolio_df.columns else None
        )
        
        return RWAResult(with_rwa, summary)


class TestRWABreakdowns(unittest.TestCase):
    """Test the enhanced RWA breakdown functionality."""

    def setUp(self):
        """Set up test data."""
        # Create a test portfolio with multiple dates, ratings and segments
        data = []
        
        for date in ['2023-12-31', '2024-03-31']:
            for rating in ['AAA', 'AA', 'A', 'BBB']:
                for segment in ['Corporate', 'Retail']:
                    # Add multiple records for each combination
                    for _ in range(5):
                        # PD increases with worse ratings
                        pd_value = {
                            'AAA': 0.01,
                            'AA': 0.05,
                            'A': 0.10,
                            'BBB': 0.15
                        }[rating] * (1.2 if segment == 'Corporate' else 0.8)
                        
                        data.append({
                            'reporting_date': date,
                            'rating': rating,
                            'segment': segment,
                            'pd': pd_value,
                            'exposure': np.random.uniform(1000, 10000)
                        })
        
        self.test_portfolio = pd.DataFrame(data)
        
        # Create a calculator with test parameters
        self.calculator = SimpleTestCalculator(regulatory_params={'confidence': 0.999})
        
    def test_rwa_calculation_with_breakdowns(self):
        """Test that RWA calculations include proper breakdowns."""
        result = self.calculator.calculate(self.test_portfolio)
        
        # Check that the result is an RWAResult
        self.assertIsInstance(result, RWAResult)
        
        # Check that basic properties work
        self.assertGreater(result.total_rwa, 0)
        self.assertGreater(result.total_exposure, 0)
        self.assertGreater(result.capital_requirement, 0)
        
        # Check for date breakdown
        self.assertTrue(result.has_breakdown('date'))
        date_breakdown = result.get_breakdown('date')
        self.assertEqual(len(date_breakdown), 2)  # Two dates
        
        # Check for rating breakdown
        self.assertTrue(result.has_breakdown('rating'))
        rating_breakdown = result.get_breakdown('rating')
        self.assertEqual(len(rating_breakdown['exposure']), 4)  # Four ratings
        
        # Check for segment breakdown
        self.assertTrue(result.has_breakdown('segment'))
        segment_breakdown = result.get_breakdown('segment')
        self.assertEqual(len(segment_breakdown['exposure']), 2)  # Two segments
        
        # Test the get_available_breakdowns method
        breakdowns = result.get_available_breakdowns()
        self.assertIn('rating', breakdowns)
        self.assertIn('segment', breakdowns)
        self.assertIn('date', breakdowns)
    
    def test_summarize_rwa_parametrization(self):
        """Test the parametrized summarize_rwa method directly."""
        # First calculate RWA
        with_rwa = self.calculator.calculate_rwa(self.test_portfolio)
        
        # Test with different breakdown parameters
        summary1 = self.calculator.summarize_rwa(with_rwa)
        summary2 = self.calculator.summarize_rwa(with_rwa, breakdown_fields=['rating'])
        summary3 = self.calculator.summarize_rwa(
            with_rwa, 
            breakdown_fields=['rating', 'segment'],
            date_field='reporting_date'
        )
        
        # Basic summary should have core metrics but no breakdowns
        self.assertIn('total_rwa', summary1)
        self.assertIn('rwa_by_rating', summary1)  # For backwards compatibility
        
        # Summary with rating breakdown should have by_rating
        self.assertIn('by_rating', summary2)
        
        # Summary with multiple breakdowns should have all specified
        self.assertIn('by_rating', summary3)
        self.assertIn('by_segment', summary3)
        self.assertIn('by_date', summary3)


if __name__ == '__main__':
    unittest.main()