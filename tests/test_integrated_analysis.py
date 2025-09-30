"""
Tests for the integrated analysis module, which connects Monte Carlo simulations with RWA calculators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Any

from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator


@dataclass
class MockRWAResult:
    """Mock result class that mimics the structure of real RWA calculator results."""
    total_rwa: float
    capital_requirement: float
    total_exposure: float
    portfolio: pd.DataFrame
    metadata: Dict[str, Any]


class MockRWACalculator:
    """Mock calculator that returns predictable results for testing."""
    
    def __init__(self, name, base_rwa_multiplier=1.0, scale_by_pd=True):
        self.name = name
        self.base_rwa_multiplier = base_rwa_multiplier
        self.scale_by_pd = scale_by_pd
    
    def calculate(self, portfolio_df):
        """Calculate mock RWA values."""
        # Use a simple formula that gives us predictable results
        if self.scale_by_pd and 'pd' in portfolio_df.columns:
            # Scale RWA by the average PD
            avg_pd = portfolio_df['pd'].mean()
            rwa_multiplier = avg_pd * 10  # Higher PD = higher RWA
        else:
            rwa_multiplier = 1.0
        
        # Calculate exposure and RWA
        total_exposure = portfolio_df['exposure'].sum() if 'exposure' in portfolio_df.columns else 1000000
        total_rwa = total_exposure * rwa_multiplier * self.base_rwa_multiplier
        
        # Create a copy of the portfolio with risk weights
        result_df = portfolio_df.copy()
        result_df['risk_weight'] = rwa_multiplier * self.base_rwa_multiplier * 100  # as percentage
        result_df['rwa'] = result_df['exposure'] * result_df['risk_weight'] / 100
        
        return MockRWAResult(
            total_rwa=total_rwa,
            capital_requirement=total_rwa * 0.08,  # 8% capital requirement
            total_exposure=total_exposure,
            portfolio=result_df,
            metadata={'calculator_type': self.name}
        )


@pytest.fixture
def sample_portfolio_data():
    """Generate a sample portfolio dataset for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Generate 50 loans over 12 quarters
    n_loans = 50
    n_quarters = 12
    
    # Generate loan IDs
    loan_ids = [f'LOAN{i:03d}' for i in range(n_loans)]
    
    # Generate dates (quarterly snapshots)
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=90*i) for i in range(n_quarters)]
    
    # Create all combinations of loan_ids and dates
    data = []
    for loan_id in loan_ids:
        for date in dates:
            # Scores between 0 and 1 with beta distribution
            score = np.random.beta(2, 5)
            
            # Generate a rating based on score
            if score < 0.05:
                rating = '1'
            elif score < 0.10:
                rating = '2'
            elif score < 0.20:
                rating = '3'
            elif score < 0.30:
                rating = '4'
            else:
                rating = '5'
            
            # Random default flag with higher probability to ensure we have some defaults
            is_default = 1 if np.random.random() < 0.05 else 0
            
            # If defaulted, set rating to 'D' and score to 1.0
            if is_default == 1:
                rating = 'D'
                score = 1.0
            
            # Generate a row
            data.append({
                'loan_id': loan_id,
                'date': date,
                'score': score,
                'rating': rating,
                'is_default': is_default,
                'into_default': 1 if is_default == 1 and date > start_date else 0,
                'exposure': np.random.uniform(50000, 500000),
                'ltv': np.random.uniform(0.4, 0.95)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def score_to_rating_bounds():
    """Define rating bounds for testing."""
    return {
        '1': (0, 0.05),
        '2': (0.05, 0.10),
        '3': (0.10, 0.20),
        '4': (0.20, 0.30),
        '5': (0.30, 1.0),
        'D': (0, 0)  # Default rating (handled separately)
    }


@pytest.fixture
def prepared_simulator(sample_portfolio_data, score_to_rating_bounds):
    """Returns a prepared simulator ready for analysis."""
    application_start_date = datetime(2021, 7, 1)
    
    simulator = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        target_auc=0.75,
        application_start_date=application_start_date,
        random_seed=42
    )
    
    simulator.prepare_simulation()
    return simulator


@pytest.fixture
def mock_calculators():
    """Create mock calculators for testing."""
    return {
        'AIRB': MockRWACalculator(name='AIRB', base_rwa_multiplier=0.8, scale_by_pd=True),
        'SA': MockRWACalculator(name='SA', base_rwa_multiplier=1.2, scale_by_pd=False)
    }


@pytest.fixture
def integrated_analysis(mock_calculators):
    """Create an IntegratedAnalysis instance with mock calculators."""
    return IntegratedAnalysis(calculators=mock_calculators)


class TestIntegratedAnalysis:
    """Tests for the IntegratedAnalysis class."""
    
    def test_initialization(self):
        """Test that the class initializes correctly."""
        analysis = IntegratedAnalysis()
        assert analysis.calculators == {}
        assert analysis.scenarios == {}
        assert analysis.results == {}
        
        # With calculators
        calculators = {'calc1': 'mock_calculator'}
        analysis = IntegratedAnalysis(calculators=calculators)
        assert analysis.calculators == calculators
    
    def test_add_calculator(self, integrated_analysis):
        """Test adding calculators."""
        # Add a new calculator
        calc = MockRWACalculator('New')
        integrated_analysis.add_calculator('New', calc)
        assert 'New' in integrated_analysis.calculators
        assert integrated_analysis.calculators['New'] == calc
        
        # Override existing calculator
        new_calc = MockRWACalculator('Updated')
        integrated_analysis.add_calculator('AIRB', new_calc)
        assert integrated_analysis.calculators['AIRB'] == new_calc
    
    def test_add_scenario(self, integrated_analysis, prepared_simulator):
        """Test adding scenarios."""
        # Add a scenario
        integrated_analysis.add_scenario(
            'Baseline',
            prepared_simulator,
            n_iterations=10,
            description="Baseline scenario"
        )
        
        assert 'Baseline' in integrated_analysis.scenarios
        assert integrated_analysis.scenarios['Baseline']['simulator'] == prepared_simulator
        assert integrated_analysis.scenarios['Baseline']['n_iterations'] == 10
        assert integrated_analysis.scenarios['Baseline']['params']['description'] == "Baseline scenario"
        
        # Add another scenario with different params
        integrated_analysis.add_scenario(
            'Improved AUC',
            prepared_simulator,
            n_iterations=20,
            target_auc=0.85,
            description="Improved model performance"
        )
        
        assert 'Improved AUC' in integrated_analysis.scenarios
        assert integrated_analysis.scenarios['Improved AUC']['n_iterations'] == 20
        assert integrated_analysis.scenarios['Improved AUC']['params']['target_auc'] == 0.85
    
    def test_run_scenario(self, integrated_analysis, prepared_simulator):
        """Test running a scenario."""
        # Add a scenario
        integrated_analysis.add_scenario(
            'Test Run',
            prepared_simulator,
            n_iterations=3,  # Use a small number for testing
            description="Test scenario"
        )
        
        # Run the scenario with all calculators
        results = integrated_analysis.run_scenario('Test Run')
        
        # Verify results structure
        assert 'Test Run' in integrated_analysis.results
        assert 'simulation_time' in results
        assert 'n_iterations' in results
        assert results['n_iterations'] == 3
        assert 'raw_simulations' in results
        assert len(results['raw_simulations']) == 3
        assert 'calculator_results' in results
        assert 'AIRB' in results['calculator_results']
        assert 'SA' in results['calculator_results']
        
        # Check calculator results
        airb_results = results['calculator_results']['AIRB']
        assert 'results' in airb_results
        assert 'calculation_time' in airb_results
        assert len(airb_results['results']) == 3
        
        # Run with only one calculator
        results2 = integrated_analysis.run_scenario('Test Run', calculator_names=['AIRB'])
        assert 'AIRB' in results2['calculator_results']
        assert 'SA' not in results2['calculator_results']
    
    def test_get_summary_stats(self, integrated_analysis, prepared_simulator):
        """Test getting summary statistics."""
        # Set up and run a scenario
        integrated_analysis.add_scenario('Stats Test', prepared_simulator, n_iterations=5)
        integrated_analysis.run_scenario('Stats Test')
        
        # Get summary stats
        stats = integrated_analysis.get_summary_stats('Stats Test', 'AIRB')
        
        # Check stats
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'count' in stats
        assert stats['count'] == 5
        
        # Check that all stats are numeric
        for key, value in stats.items():
            if key != 'count':
                assert isinstance(value, float)
    
    def test_get_percentiles(self, integrated_analysis, prepared_simulator):
        """Test getting percentiles from results."""
        # Set up and run a scenario
        integrated_analysis.add_scenario('Percentile Test', prepared_simulator, n_iterations=10)
        integrated_analysis.run_scenario('Percentile Test')
        
        # Get percentiles
        percentiles = integrated_analysis.get_percentiles(
            'Percentile Test', 
            'AIRB', 
            percentiles=(25, 50, 75)
        )
        
        # Check percentiles
        assert 25 in percentiles
        assert 50 in percentiles
        assert 75 in percentiles
        assert percentiles[25] <= percentiles[50] <= percentiles[75]
    
    def test_compare_scenarios(self, integrated_analysis, prepared_simulator):
        """Test comparing multiple scenarios."""
        # Create two scenarios with different parameters
        simulator1 = prepared_simulator
        
        # Clone the simulator with a different target AUC
        simulator2 = PortfolioSimulator(
            portfolio_df=simulator1.portfolio_df.copy(),
            score_to_rating_bounds=simulator1.score_to_rating_bounds,
            rating_col=simulator1.rating_col,
            loan_id_col=simulator1.loan_id_col,
            date_col=simulator1.date_col,
            default_col=simulator1.default_col,
            into_default_flag_col=simulator1.into_default_flag_col,
            score_col=simulator1.score_col,
            target_auc=0.85,  # Higher than simulator1
            application_start_date=simulator1.application_start_date,
            random_seed=43  # Different seed
        )
        simulator2.prepare_simulation()
        
        # Add scenarios
        integrated_analysis.add_scenario('Baseline', simulator1, n_iterations=5)
        integrated_analysis.add_scenario('Improved', simulator2, n_iterations=5)
        
        # Run scenarios
        integrated_analysis.run_scenario('Baseline')
        integrated_analysis.run_scenario('Improved')
        
        # Compare scenarios
        comparison = integrated_analysis.compare_scenarios(['Baseline', 'Improved'], 'AIRB')
        
        # Check comparison DataFrame
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'scenario' in comparison.columns
        assert 'mean' in comparison.columns
        assert 'abs_diff_from_baseline' in comparison.columns
        assert 'pct_diff_from_baseline' in comparison.columns
        
        # First row should be baseline with no difference
        assert comparison.iloc[0]['scenario'] == 'Baseline'
        assert pd.isna(comparison.iloc[0]['abs_diff_from_baseline']) if 'abs_diff_from_baseline' in comparison.iloc[0] else True
        
        # Second row should be improved with some difference
        assert comparison.iloc[1]['scenario'] == 'Improved'
        assert 'abs_diff_from_baseline' in comparison.iloc[1]
    
    def test_get_rwa_distribution(self, integrated_analysis, prepared_simulator):
        """Test getting the full RWA distribution."""
        # Set up and run a scenario
        integrated_analysis.add_scenario('Distribution Test', prepared_simulator, n_iterations=10)
        integrated_analysis.run_scenario('Distribution Test')
        
        # Get distribution
        distribution = integrated_analysis.get_rwa_distribution('Distribution Test', 'AIRB')
        
        # Check distribution
        assert isinstance(distribution, pd.Series)
        assert len(distribution) == 10
        assert distribution.name == 'total_rwa'
        
        # Check that values are numeric and positive
        assert distribution.dtype == float
        assert (distribution > 0).all()
    
    def test_error_handling(self, integrated_analysis):
        """Test error handling for invalid scenarios and calculators."""
        # Test non-existent scenario
        with pytest.raises(ValueError, match="does not exist"):
            integrated_analysis.run_scenario('NonExistent')
        
        # Test non-existent calculator
        with pytest.raises(ValueError, match="does not exist"):
            integrated_analysis.run_scenario('Test', calculator_names=['NonExistent'])
        
        # Test invalid scenario name in summary stats
        with pytest.raises(ValueError, match="No results for scenario"):
            integrated_analysis.get_summary_stats('NonExistent', 'AIRB')
        
        # Test invalid calculator name in summary stats
        integrated_analysis.add_scenario('ErrorTest', None)  # Just add a placeholder
        integrated_analysis.results['ErrorTest'] = {'calculator_results': {}}  # Mock partial results
        with pytest.raises(ValueError, match="No results for calculator"):
            integrated_analysis.get_summary_stats('ErrorTest', 'NonExistent')


def test_integrated_analysis_end_to_end(sample_portfolio_data, score_to_rating_bounds):
    """End-to-end test simulating a real-world usage scenario."""
    # Create simulators with different AUC targets
    application_start_date = datetime(2021, 7, 1)
    
    base_simulator = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        target_auc=0.75,  # Baseline AUC
        application_start_date=application_start_date,
        random_seed=42
    )
    base_simulator.prepare_simulation()
    
    improved_simulator = PortfolioSimulator(
        portfolio_df=sample_portfolio_data,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        target_auc=0.85,  # Improved AUC
        application_start_date=application_start_date,
        random_seed=42
    )
    improved_simulator.prepare_simulation()
    
    # Create calculators
    airb_calculator = MockRWACalculator('AIRB', base_rwa_multiplier=0.8, scale_by_pd=True)
    sa_calculator = MockRWACalculator('SA', base_rwa_multiplier=1.2, scale_by_pd=False)
    
    # Create integrated analysis
    analysis = IntegratedAnalysis()
    analysis.add_calculator('AIRB', airb_calculator)
    analysis.add_calculator('SA', sa_calculator)
    
    # Add scenarios
    analysis.add_scenario('Baseline AUC 0.75', base_simulator, n_iterations=5)
    analysis.add_scenario('Improved AUC 0.85', improved_simulator, n_iterations=5)
    
    # Run scenarios
    analysis.run_scenario('Baseline AUC 0.75')
    analysis.run_scenario('Improved AUC 0.85')
    
    # Get summary stats
    baseline_stats = analysis.get_summary_stats('Baseline AUC 0.75', 'AIRB')
    improved_stats = analysis.get_summary_stats('Improved AUC 0.85', 'AIRB')
    
    # Compare scenarios
    comparison = analysis.compare_scenarios(
        ['Baseline AUC 0.75', 'Improved AUC 0.85'], 
        'AIRB'
    )
    
    # Get percentiles
    baseline_percentiles = analysis.get_percentiles('Baseline AUC 0.75', 'AIRB')
    improved_percentiles = analysis.get_percentiles('Improved AUC 0.85', 'AIRB')
    
    # Print results (for debugging)
    print("\nEnd-to-End Test Results:")
    print(f"Baseline AIRB mean RWA: {baseline_stats['mean']:,.2f}")
    print(f"Improved AIRB mean RWA: {improved_stats['mean']:,.2f}")
    print(f"Percentage difference: {(improved_stats['mean']/baseline_stats['mean'] - 1)*100:.2f}%")
    
    # Because of how our mock calculator works (higher PD = higher RWA),
    # and better AUC means better separation between good and bad,
    # the improved model should generally lead to different RWA
    # The exact direction depends on the portfolio mix
    
    # Check that we got valid results
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 2
    assert 'pct_diff_from_baseline' in comparison.columns
    assert baseline_stats['mean'] > 0
    assert improved_stats['mean'] > 0