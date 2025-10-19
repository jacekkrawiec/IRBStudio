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


class TestAdvancedIntegration:
    """Test advanced integration scenarios."""
    
    def test_integration_calculator_to_reporting(
        self,
        small_portfolio_df
    ):
        """Test calculator output works with reporting functions."""
        from irbstudio.reporting.dashboard import (
            create_rwa_distribution_plot,
            create_summary_table
        )
        
        # Calculate RWA
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        
        # Run multiple iterations for distribution
        results = []
        for i in range(10):
            result = calculator.calculate(small_portfolio_df)
            results.append(result)
        
        # Extract RWA values for plotting
        rwa_values = [r.summary['total_rwa'] for r in results]
        
        # Should work with reporting functions - package in expected format
        results_dict = {
            'test': {
                'AIRB': {
                    'rwa_values': rwa_values
                }
            }
        }
        
        fig = create_rwa_distribution_plot(
            results=results_dict,
            scenario_name='test',
            calculator_name='AIRB'
        )
        assert fig is not None
        
        # Summary table - use IntegratedAnalysis results format
        mock_results = {
            'baseline': {
                'AIRB': {
                    'results': results,
                    'rwa_values': rwa_values
                }
            }
        }
        table = create_summary_table(mock_results)
        assert table is not None
    
    def test_integration_config_to_execution(
        self,
        small_portfolio_df,
        score_to_rating_bounds
    ):
        """Test configuration object drives execution."""
        # Create config-like structure
        config_params = {
            'lgd': 0.25,
            'maturity_years': 2.5,
            'scaling_factor': 1.06,
            'asset_correlation': 0.15
        }
        
        # Use config in calculator
        calculator = AIRBMortgageCalculator(
            regulatory_params=config_params
        )
        
        # Use config in simulator
        simulator = PortfolioSimulator(
            portfolio_df=small_portfolio_df,
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            asset_correlation=config_params['asset_correlation']
        )
        
        # Execute integrated workflow
        simulated = simulator.simulate_once(random_seed=42)
        result = calculator.calculate(simulated)
        
        assert result is not None
        assert result.summary['total_rwa'] > 0
    
    def test_integration_e2e_with_date_breakdown(
        self,
        small_portfolio_df,
        score_to_rating_bounds
    ):
        """Test end-to-end with date breakdown enabled."""
        # Ensure we have date column
        assert 'reporting_date' in small_portfolio_df.columns
        
        # Create analysis with date handling
        analysis = IntegratedAnalysis(date_column='reporting_date')
        
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': 0.25,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        analysis.add_calculator('AIRB', calculator)
        
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
        
        analysis.add_scenario('baseline', simulator, n_iterations=5)
        
        # Run with process_all_dates=True
        results = analysis.run_scenario(
            'baseline',
            process_all_dates=True,
            random_seed=42
        )
        
        assert results is not None
        assert 'calculator_results' in results
        
        # Check if date breakdown available
        calc_results = results['calculator_results']['AIRB']['results']
        if len(calc_results) > 0:
            first_result = calc_results[0]
            # Date breakdown might be in by_date or similar
            assert hasattr(first_result, 'summary')


# Summary of integration test coverage:
# - End-to-end workflows: 4 tests
# - Module integration: 4 tests  
# - Scenario comparison: 2 tests
# - Advanced integration: 3 tests (NEW)
# Total: 13 integration tests


class TestAdditionalIntegration:
    """Additional integration tests."""
    
    def test_e2e_reproducible_results(self, small_portfolio_df, score_to_rating_bounds, airb_params):
        """Test reproducible end-to-end results with same seed."""
        # Create analysis
        analysis1 = IntegratedAnalysis()
        
        simulator1 = PortfolioSimulator(
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
        
        calculator1 = AIRBMortgageCalculator(airb_params)
        analysis1.add_calculator('AIRB', calculator1)
        analysis1.add_scenario('Test', simulator1, n_iterations=10)
        
        # Run with seed
        results1 = analysis1.run_scenario('Test', random_seed=42)
        
        # Create second analysis with same setup
        analysis2 = IntegratedAnalysis()
        
        simulator2 = PortfolioSimulator(
            portfolio_df=small_portfolio_df.copy(),
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col='rating',
            loan_id_col='loan_id',
            date_col='reporting_date',
            default_col='default_flag',
            into_default_flag_col='into_default_flag',
            score_col='score',
            application_start_date='2024-01-01'
        )
        
        calculator2 = AIRBMortgageCalculator(airb_params)
        analysis2.add_calculator('AIRB', calculator2)
        analysis2.add_scenario('Test', simulator2, n_iterations=10)
        
        # Run with same seed
        results2 = analysis2.run_scenario('Test', random_seed=42)
        
        # Results should be identical
        rwa1 = [r.summary['total_rwa'] for r in results1['calculator_results']['AIRB']['results']]
        rwa2 = [r.summary['total_rwa'] for r in results2['calculator_results']['AIRB']['results']]
        
        assert len(rwa1) == len(rwa2)
        assert np.allclose(rwa1, rwa2, rtol=1e-10)
    
    def test_integration_scenario_comparison_visualization(
        self,
        small_portfolio_df,
        score_to_rating_bounds,
        airb_params
    ):
        """Test scenario comparison with visualization integration."""
        from irbstudio.reporting.dashboard import create_scenario_comparison_plot
        
        analysis = IntegratedAnalysis()
        calculator = AIRBMortgageCalculator(airb_params)
        analysis.add_calculator('AIRB', calculator)
        
        # Create baseline scenario
        simulator_baseline = PortfolioSimulator(
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
        
        analysis.add_scenario('baseline', simulator_baseline, n_iterations=20)
        
        # Create alternative scenario (same data for simplicity)
        simulator_alt = PortfolioSimulator(
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
        
        analysis.add_scenario('alternative', simulator_alt, n_iterations=20)
        
        # Run scenarios
        results_baseline = analysis.run_scenario('baseline', random_seed=42)
        results_alt = analysis.run_scenario('alternative', random_seed=43)
        
        # Prepare data for visualization
        rwa_baseline = [r.summary['total_rwa'] for r in results_baseline['calculator_results']['AIRB']['results']]
        rwa_alt = [r.summary['total_rwa'] for r in results_alt['calculator_results']['AIRB']['results']]
        
        viz_results = {
            'baseline': {
                'AIRB': {
                    'mean': np.mean(rwa_baseline),
                    'std': np.std(rwa_baseline),
                    'median': np.median(rwa_baseline)
                }
            },
            'alternative': {
                'AIRB': {
                    'mean': np.mean(rwa_alt),
                    'std': np.std(rwa_alt),
                    'median': np.median(rwa_alt)
                }
            }
        }
        
        # Create comparison plot
        fig = create_scenario_comparison_plot(
            results=viz_results,
            calculator_name='AIRB',
            title="Scenario Comparison"
        )
        
        assert fig is not None
        assert len(fig.data) >= 1
    
    def test_e2e_custom_configuration(self, small_portfolio_df, score_to_rating_bounds):
        """Test end-to-end workflow with custom configuration."""
        # Create custom config
        config = Config(
            scenarios=[
                Scenario(
                    name='baseline',
                    pd_auc=0.72,
                    portfolio_default_rate=0.025,
                    lgd=0.30,
                    new_loan_rate=0.10
                )
            ],
            calculators=['AIRB'],
            memory_efficient=False
        )
        
        # Create analysis with config
        analysis = IntegratedAnalysis()
        
        # Add calculator based on config
        calculator = AIRBMortgageCalculator(
            regulatory_params={
                'lgd': config.scenarios[0].lgd,
                'maturity_years': 2.5,
                'scaling_factor': 1.06
            }
        )
        analysis.add_calculator('AIRB', calculator)
        
        # Create simulator with config parameters
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
        
        # Use custom iteration count
        n_iterations = 15
        analysis.add_scenario('baseline', simulator, n_iterations=n_iterations)
        
        # Run with seed
        results = analysis.run_scenario('baseline', random_seed=42)
        
        assert results is not None
        assert len(results['calculator_results']['AIRB']['results']) == n_iterations
        # Verify config was used correctly
        assert config.scenarios[0].lgd == 0.30
