"""
Integration tests for IRBStudio.

Priority 3: Integration Tests (18 tests)
- End-to-end workflows
- Module integration
- Scenario comparison workflows
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.engine.mortgage import AIRBMortgageCalculator, SAMortgageCalculator
from irbstudio.data.loader import load_portfolio
from irbstudio.config.schema import Config, Scenario, ColumnMapping


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_e2e_complete_analysis(self, small_portfolio_df, score_to_rating_bounds):
        """Test complete analysis from portfolio to results."""
        # 1. Create simulator
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
        
        # 2. Create analysis
        analysis = IntegratedAnalysis(date_column='reporting_date')
        
        # 3. Add calculator
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        analysis.add_calculator('AIRB', calculator)
        
        # 4. Add scenario
        analysis.add_scenario('baseline', simulator, n_iterations=3)
        
        # 5. Run analysis
        results = analysis.run_scenario('baseline', random_seed=42)
        
        # 6. Verify results
        assert 'calculator_results' in results
        assert 'AIRB' in results['calculator_results']
        assert len(results['calculator_results']['AIRB']['results']) == 3
        
        # 7. Check summary statistics
        for result in results['calculator_results']['AIRB']['results']:
            assert 'total_rwa' in result.summary
            assert 'total_exposure' in result.summary
            assert result.summary['total_rwa'] > 0
    
    def test_e2e_csv_to_results(self, small_portfolio_df):
        """Test workflow from CSV file to results."""
        # 1. Save portfolio to CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            small_portfolio_df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # 2. Load portfolio
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='exposure'
            )
            df = load_portfolio(temp_path, mapping)
            
            assert df is not None
            assert len(df) > 0
            assert 'loan_id' in df.columns
            assert 'exposure' in df.columns
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_e2e_multiple_scenarios(self, small_portfolio_df, score_to_rating_bounds):
        """Test multiple scenarios in full workflow."""
        analysis = IntegratedAnalysis(date_column='reporting_date')
        
        # Add calculator
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        analysis.add_calculator('AIRB', calculator)
        
        # Add multiple scenarios with different simulators
        sim_baseline = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score'
        )
        
        sim_stressed = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score'
        )
        
        analysis.add_scenario('baseline', sim_baseline, n_iterations=2)
        analysis.add_scenario('stressed', sim_stressed, n_iterations=2)
        
        # Run both scenarios
        results_baseline = analysis.run_scenario('baseline', random_seed=42)
        results_stressed = analysis.run_scenario('stressed', random_seed=43)
        
        # Verify both completed
        assert len(results_baseline['calculator_results']['AIRB']['results']) == 2
        assert len(results_stressed['calculator_results']['AIRB']['results']) == 2
    
    def test_e2e_both_calculators(self, small_portfolio_df, score_to_rating_bounds):
        """Test AIRB and SA calculators together."""
        analysis = IntegratedAnalysis(date_column='reporting_date')
        
        # Add both calculators
        airb = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        sa = SAMortgageCalculator(
            regulatory_params={}
        )
        
        analysis.add_calculator('AIRB', airb)
        analysis.add_calculator('SA', sa)
        
        # Add scenario
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
        analysis.add_scenario('test', simulator, n_iterations=2)
        
        # Run with both calculators
        results = analysis.run_scenario('test', random_seed=42)
        
        # Both should have results
        assert 'AIRB' in results['calculator_results']
        assert 'SA' in results['calculator_results']
        assert len(results['calculator_results']['AIRB']['results']) == 2
        assert len(results['calculator_results']['SA']['results']) == 2


class TestModuleIntegration:
    """Test integration between different modules."""
    
    def test_integration_simulator_to_calculator(
        self,
        small_portfolio_df,
        score_to_rating_bounds
    ):
        """Test simulator output works with calculator input."""
        # Create and run simulator
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
        
        simulated_df = simulator.simulate_once(random_seed=42)
        
        # Use simulated output in calculator
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        
        # Calculator should accept simulated portfolio
        result = calculator.calculate(simulated_df)
        
        assert result is not None
        assert hasattr(result, 'summary')
        assert 'total_rwa' in result.summary
    
    def test_integration_data_loader_to_simulator(
        self,
        small_portfolio_df,
        score_to_rating_bounds
    ):
        """Test data loader output works with simulator input."""
        # Save and reload through data loader
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            small_portfolio_df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='exposure'
            )
            loaded_df = load_portfolio(temp_path, mapping)
            
            # Loaded data should work with simulator
            simulator = PortfolioSimulator(
                portfolio_df=loaded_df,
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
            assert len(result) > 0
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_integration_multiple_calculators(self, small_portfolio_df):
        """Test multiple calculators on same portfolio."""
        # Create both calculators
        airb = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        sa = SAMortgageCalculator(
            regulatory_params={}
        )
        
        # Both should process same portfolio
        result_airb = airb.calculate(small_portfolio_df)
        result_sa = sa.calculate(small_portfolio_df)
        
        assert result_airb is not None
        assert result_sa is not None
        assert 'total_rwa' in result_airb.summary
        assert 'total_rwa' in result_sa.summary
        
        # Results should differ (different methodologies)
        assert result_airb.summary['total_rwa'] != result_sa.summary['total_rwa']
    
    def test_integration_column_mapping_throughout(self, small_portfolio_df):
        """Test column mapping works across entire pipeline."""
        # Rename columns
        df_renamed = small_portfolio_df.rename(columns={
            'loan_id': 'LOAN_NUMBER',
            'exposure': 'BALANCE'
        })
        
        # Save with renamed columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_renamed.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Load with mapping
            mapping = ColumnMapping(
                loan_id='LOAN_NUMBER',
                exposure='BALANCE'
            )
            loaded_df = load_portfolio(temp_path, mapping)
            
            # Should have canonical names
            assert 'loan_id' in loaded_df.columns
            assert 'exposure' in loaded_df.columns
            
            # Should work with calculator
            calculator = AIRBMortgageCalculator(
                regulatory_params={
                    'lgd': 0.25,
                    'maturity_years': 2.5,
                    'scaling_factor': 1.06
                }
            )
            result = calculator.calculate(loaded_df)
            assert result is not None
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestScenarioComparisonIntegration:
    """Test scenario comparison workflows."""
    
    def test_integration_scenario_comparison_workflow(
        self,
        small_portfolio_df,
        score_to_rating_bounds
    ):
        """Test complete scenario comparison workflow."""
        analysis = IntegratedAnalysis(date_column='reporting_date')
        
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        analysis.add_calculator('AIRB', calculator)
        
        # Create two scenarios
        sim1 = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score'
        )
        
        sim2 = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score'
        )
        
        analysis.add_scenario('scenario1', sim1, n_iterations=5)
        analysis.add_scenario('scenario2', sim2, n_iterations=5)
        
        # Run both
        results1 = analysis.run_scenario('scenario1', random_seed=42)
        results2 = analysis.run_scenario('scenario2', random_seed=100)
        
        # Both should complete
        assert results1 is not None
        assert results2 is not None
        
        # Extract RWA values for comparison
        rwa1 = [r.summary['total_rwa'] for r in results1['calculator_results']['AIRB']['results']]
        rwa2 = [r.summary['total_rwa'] for r in results2['calculator_results']['AIRB']['results']]
        
        assert len(rwa1) == 5
        assert len(rwa2) == 5
        
        # Can calculate statistics
        mean_rwa1 = np.mean(rwa1)
        mean_rwa2 = np.mean(rwa2)
        
        assert mean_rwa1 > 0
        assert mean_rwa2 > 0
    
    def test_integration_scenario_comparison_capital_delta(
        self,
        small_portfolio_df,
        score_to_rating_bounds
    ):
        """Test capital delta calculation between scenarios."""
        analysis = IntegratedAnalysis(date_column='reporting_date')
        
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        analysis.add_calculator('AIRB', calculator)
        
        # Two scenarios
        sim1 = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score'
        )
        
        sim2 = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score'
        )
        
        analysis.add_scenario('base', sim1, n_iterations=3)
        analysis.add_scenario('alt', sim2, n_iterations=3)
        
        results_base = analysis.run_scenario('base', random_seed=42)
        results_alt = analysis.run_scenario('alt', random_seed=100)
        
        # Calculate delta
        rwa_base = [r.summary['total_rwa'] for r in results_base['calculator_results']['AIRB']['results']]
        rwa_alt = [r.summary['total_rwa'] for r in results_alt['calculator_results']['AIRB']['results']]
        
        mean_delta = np.mean(rwa_alt) - np.mean(rwa_base)
        
        # Delta can be positive, negative, or zero
        assert isinstance(mean_delta, (int, float))
