"""
Tests for the integrated analysis module.

Priority 1: Critical - Core Functionality
"""

import pytest
import pandas as pd
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


class TestIntegratedAnalysisRunScenarioVariations:
    """Tests for IntegratedAnalysis.run_scenario() parameter variations."""
    
    def test_integrated_analysis_run_scenario_single_calculator(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with single calculator specified."""
        analysis = IntegratedAnalysis()
        
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'  # Historical: 2023-07 to 2023-12
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=5)
        
        # Add multiple calculators but only use one
        calc1 = AIRBMortgageCalculator(airb_params)
        calc2 = AIRBMortgageCalculator(airb_params)
        
        analysis.add_calculator('AIRB1', calc1)
        analysis.add_calculator('AIRB2', calc2)
        
        # Run with only one calculator
        results = analysis.run_scenario('Test', calculator_names=['AIRB1'])
        
        assert 'calculator_results' in results
        assert 'AIRB1' in results['calculator_results']
        assert 'AIRB2' not in results['calculator_results']
    
    def test_integrated_analysis_run_scenario_multiple_calculators(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with multiple calculators."""
        analysis = IntegratedAnalysis()
        
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=5)
        
        calc1 = AIRBMortgageCalculator(airb_params)
        calc2 = AIRBMortgageCalculator(airb_params)
        
        analysis.add_calculator('AIRB1', calc1)
        analysis.add_calculator('AIRB2', calc2)
        
        # Run with both calculators
        results = analysis.run_scenario('Test', calculator_names=['AIRB1', 'AIRB2'])
        
        assert 'calculator_results' in results
        assert 'AIRB1' in results['calculator_results']
        assert 'AIRB2' in results['calculator_results']
        assert len(results['calculator_results']['AIRB1']['results']) == 5
        assert len(results['calculator_results']['AIRB2']['results']) == 5
    
    def test_integrated_analysis_run_scenario_memory_efficient(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with memory_efficient=True."""
        analysis = IntegratedAnalysis()
        
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=5)
        
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Run with memory efficient mode
        results = analysis.run_scenario('Test', memory_efficient=True)
        
        assert results is not None
        assert 'calculator_results' in results
        assert len(results['calculator_results']['AIRB']['results']) == 5
    
    def test_integrated_analysis_run_scenario_standard_mode(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with memory_efficient=False."""
        analysis = IntegratedAnalysis()
        
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=5)
        
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Run with standard mode (stores all DataFrames)
        results = analysis.run_scenario('Test', memory_efficient=False)
        
        assert results is not None
        assert 'calculator_results' in results
        assert len(results['calculator_results']['AIRB']['results']) == 5
    
    def test_integrated_analysis_run_scenario_portfolio_filter(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with custom portfolio filter function."""
        analysis = IntegratedAnalysis()
        
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=5)
        
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Define a filter function (only high exposure loans)
        def filter_high_exposure(df):
            return df[df['exposure'] > df['exposure'].median()]
        
        # Run with filter
        results = analysis.run_scenario('Test', portfolio_filter=filter_high_exposure)
        
        assert results is not None
        assert 'calculator_results' in results
        # Results should be based on filtered portfolio
        assert len(results['calculator_results']['AIRB']['results']) == 5
    
    def test_integrated_analysis_run_scenario_store_full_portfolio(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with store_full_portfolio=True."""
        analysis = IntegratedAnalysis()
        
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=5)
        
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Run with full portfolio storage
        results = analysis.run_scenario('Test', store_full_portfolio=True)
        
        assert results is not None
        assert 'calculator_results' in results
        # Check that results contain portfolio data
        rwa_results = results['calculator_results']['AIRB']['results']
        assert len(rwa_results) == 5
    
    def test_integrated_analysis_run_scenario_missing_calculator(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with non-existent calculator name."""
        analysis = IntegratedAnalysis()
        
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=5)
        
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Try to run with non-existent calculator
        with pytest.raises(ValueError, match="Calculator 'NonExistent' does not exist"):
            analysis.run_scenario('Test', calculator_names=['NonExistent'])
    
    def test_integrated_analysis_run_scenario_missing_scenario(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with non-existent scenario name."""
        analysis = IntegratedAnalysis()
        
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Try to run non-existent scenario
        with pytest.raises(ValueError, match="Scenario 'NonExistent' does not exist"):
            analysis.run_scenario('NonExistent')
    
    def test_integrated_analysis_run_scenario_column_renaming(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with exposure column rename."""
        # Create copy with custom PD column name
        custom_df = small_portfolio_df.copy()
        custom_df['probability_of_default'] = custom_df['pd']
        custom_df.drop(columns=['pd'], inplace=True)
        
        # Create analysis with custom column mapping
        analysis = IntegratedAnalysis(
            pd_column='probability_of_default',
            target_pd_column='pd'  # Will rename to 'pd' for calculators
        )
        
        simulator = PortfolioSimulator(
            portfolio_df=custom_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=3)
        
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Run scenario - should handle column renaming
        results = analysis.run_scenario('Test')
        
        assert results is not None
        assert 'calculator_results' in results
        assert len(results['calculator_results']['AIRB']['results']) == 3
    
    def test_integrated_analysis_run_scenario_progress_tracking(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params,
        caplog
    ):
        """Test run_scenario logs progress updates."""
        import logging
        caplog.set_level(logging.INFO)
        
        analysis = IntegratedAnalysis()
        
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=10)
        
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Run scenario - should log progress
        results = analysis.run_scenario('Test', memory_efficient=True)
        
        assert results is not None
        
        # Check for progress logs (every 5 iterations + final)
        log_text = caplog.text
        assert '[CHECKPOINT]' in log_text or 'Starting scenario' in log_text
        assert '[PROGRESS]' in log_text or 'Completed' in log_text
    
    def test_integrated_analysis_run_scenario_process_all_dates(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test run_scenario with process_all_dates parameter (verifies parameter is accepted)."""
        analysis = IntegratedAnalysis()
        
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        analysis.add_scenario('Test', simulator, n_iterations=3)
        
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Run with process_all_dates=True (verifies parameter is accepted)
        # For single-date portfolio, behavior should be same as False
        results = analysis.run_scenario('Test', process_all_dates=True)
        
        assert results is not None
        assert 'calculator_results' in results
        assert len(results['calculator_results']['AIRB']['results']) == 3
