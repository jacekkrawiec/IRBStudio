"""
Example showing how to use the enhanced RWA breakdown capabilities.

This example demonstrates:
1. Creating an IntegratedAnalysis instance with custom column mapping
2. Processing simulations with the process_all_dates parameter
3. Using the new breakdown features in RWAResult
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import the necessary classes
from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.engine.base import BaseRWACalculator, RWAResult


# Simple RWA calculator for demonstration
class SimpleRWACalculator(BaseRWACalculator):
    """Simple calculator implementation for the example."""
    
    def calculate_rw(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk weights using a simple PD-based formula."""
        df = portfolio_df.copy()
        
        # Simple formula: RW = PD * 12.5 (typical multiplier in Basel)
        # Cap risk weight at 150%
        df['risk_weight'] = df['pd'].apply(lambda x: min(1.5, x * 12.5))
        
        return df

    def calculate_rwa(self, portfolio_df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """Calculate RWA by multiplying exposure by risk weight."""
        df = self.calculate_rw(portfolio_df)
        df['rwa'] = df['exposure'] * df['risk_weight']
        return df
    
    def calculate(self, portfolio_df: pd.DataFrame) -> RWAResult:
        """Calculate RWA and return a RWAResult object with breakdowns."""
        required_cols = ['pd', 'exposure']
        self.validate_inputs(portfolio_df, required_cols)
        
        with_rwa = self.calculate_rwa(portfolio_df)
        
        # Use our enhanced summarize_rwa method with breakdowns
        summary = self.summarize_rwa(
            with_rwa,
            breakdown_fields=['rating', 'segment'],
            date_field='date' if 'date' in portfolio_df.columns else None
        )
        
        metadata = {
            'calculator_type': 'SimpleRWA',
            'calculation_time': datetime.now().isoformat()
        }
        
        return RWAResult(with_rwa, summary, metadata)


def main():
    """Run the example."""
    # Create a sample portfolio with multiple dates and segments
    portfolio_data = []
    
    # Create 3 dates with declining PDs (improving portfolio)
    dates = [
        (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
        (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        datetime.now().strftime('%Y-%m-%d')
    ]
    
    # We'll use "date" as our date column instead of "reporting_date"
    # to demonstrate custom column mapping
    for date in dates:
        for rating in ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']:
            for segment in ['Corporate', 'SME', 'Retail']:
                # Add multiple loans for each combination
                for _ in range(10):
                    # PD increases with worse ratings, decreases over time
                    pd_base = {
                        'AAA': 0.0005,
                        'AA': 0.001,
                        'A': 0.005,
                        'BBB': 0.01,
                        'BB': 0.03,
                        'B': 0.06
                    }[rating]
                    
                    # Segment adjustments
                    segment_factor = {
                        'Corporate': 1.0,
                        'SME': 0.9,      # SMEs have slightly lower PDs
                        'Retail': 0.7    # Retail has even lower PDs
                    }[segment]
                    
                    # Time improvement - each date is 10% better
                    date_index = dates.index(date)
                    time_factor = 1.0 - (date_index * 0.1)
                    
                    # Final PD calculation
                    pd_value = pd_base * segment_factor * time_factor
                    
                    portfolio_data.append({
                        'id': f"LOAN-{len(portfolio_data) + 1}",
                        'date': date,
                        'rating': rating,
                        'segment': segment,
                        'simulated_pd': pd_value,  # Using simulated_pd instead of pd
                        'exposure': np.random.uniform(1000000, 5000000)
                    })
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame(portfolio_data)
    print(f"Created portfolio with {len(portfolio_df)} exposures")
    
    # Create a simple portfolio simulator
    simulator = PortfolioSimulator(portfolio_df)
    simulator.set_pd_simulation_params(shock_mean=0.0, shock_std=0.2)
    
    # Create an RWA calculator
    calculator = SimpleRWACalculator(regulatory_params={'confidence': 0.999})
    
    # Create an IntegratedAnalysis instance with custom column mapping
    analysis = IntegratedAnalysis(
        calculators={'SimpleRWA': calculator},
        date_column='date',           # Use 'date' instead of default 'reporting_date'
        pd_column='simulated_pd',     # Column containing PD values
        target_pd_column='pd'         # Column name expected by calculator
    )
    
    # Add a scenario
    analysis.add_scenario(
        name='baseline', 
        simulator=simulator,
        n_iterations=50
    )
    
    # Run the scenario with process_all_dates=True to analyze all dates
    results = analysis.run_scenario(
        scenario_name='baseline',
        calculator_names=['SimpleRWA'],
        process_all_dates=True,
        memory_efficient=True
    )
    
    # Get summary statistics
    stats = analysis.get_summary_stats('baseline', 'SimpleRWA')
    print("\nSummary Statistics:")
    for stat, value in stats.items():
        print(f"  {stat}: {value:,.2f}")
    
    # Get the first result to examine breakdowns
    calculator_results = analysis._get_calculator_results('baseline', 'SimpleRWA')
    if calculator_results:
        result = calculator_results[0]
        
        print("\nAvailable Breakdowns:", result.get_available_breakdowns())
        
        # Print segment breakdown
        if result.has_breakdown('segment'):
            segment_data = result.get_breakdown('segment')
            print("\nRWA by Segment:")
            for segment, rwa in segment_data['rwa'].items():
                exposure = segment_data['exposure'][segment]
                print(f"  {segment}: RWA = {rwa:,.2f}, Exposure = {exposure:,.2f}")
        
        # Print date breakdown
        if result.has_breakdown('date'):
            date_data = result.get_breakdown('date')
            print("\nRWA by Date:")
            for date, rwa in date_data['rwa'].items():
                exposure = date_data['exposure'][date]
                print(f"  {date}: RWA = {rwa:,.2f}, Exposure = {exposure:,.2f}")
        
        # Plot rating breakdown if available
        if result.has_breakdown('rating'):
            rating_data = result.get_breakdown('rating')
            ratings = list(rating_data['rwa'].keys())
            rwas = [rating_data['rwa'][r] for r in ratings]
            
            plt.figure(figsize=(10, 6))
            plt.bar(ratings, rwas)
            plt.title('RWA by Rating')
            plt.xlabel('Rating')
            plt.ylabel('RWA')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()