"""
Tests for the integrated analysis module.

Priority 1: Critical - Core Functionality
"""

import pytest
from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.engine.mortgage.airb_calculator import AIRBMortgageCalculator


class TestIntegratedAnalysis:
    """Tests for IntegratedAnalysis basic functionality."""
    
    def test_integrated_analysis_run_scenario_basic(
        self, 
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test IntegratedAnalysis.run_scenario() with basic setup."""
        analysis = IntegratedAnalysis()
        
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
        
        analysis.add_scenario('Baseline', simulator, n_iterations=2)
        
        calculator = AIRBMortgageCalculator(airb_params)
        
        analysis.add_calculator('AIRB', calculator)
        
        results = analysis.run_scenario('Baseline')
        
        assert results is not None
        assert 'calculator_results' in results
        assert 'AIRB' in results['calculator_results']
    
    def test_integrated_analysis_with_config(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        sample_config_dict,
        airb_params
    ):
        """Test IntegratedAnalysis with configuration."""
        analysis = IntegratedAnalysis()
        
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
        
        analysis.add_scenario('Test', simulator, n_iterations=2)
        
        calculator = AIRBMortgageCalculator(airb_params)
        
        analysis.add_calculator('AIRB', calculator)
        
        results = analysis.run_scenario('Test')
        
        assert results is not None
