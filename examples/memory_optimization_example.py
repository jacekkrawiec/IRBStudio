# Example demonstrating memory optimization in IntegratedAnalysis
"""
This example demonstrates two memory optimization strategies in the IRBStudio framework:

1. Memory-Efficient Processing:
   - Process one Monte Carlo iteration at a time instead of generating all iterations upfront
   - Controlled by the `memory_efficient` parameter in IntegratedAnalysis.run_scenario()
   - Best for large numbers of iterations (10+) or very large portfolios

2. Memory-Efficient Storage:
   - Store only essential columns in RWAResult objects (rwa, risk_weight, exposure) 
   - Controlled by the `store_full_portfolio` parameter in IntegratedAnalysis.run_scenario()
   - Best when full portfolio data is not needed for post-processing

Memory Optimization Benefits:
- Reduced memory footprint for large datasets (10M+ rows)
- Ability to run more iterations with limited memory
- Faster processing due to better cache utilization
- Less garbage collection overhead

The example compares four scenarios:
1. No optimization
2. Memory-efficient processing only
3. Memory-efficient storage only
4. Full optimization (both strategies)

Results show memory usage before and after running simulations, including peak memory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
from datetime import datetime
import psutil

# Import IRBStudio classes
from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.engine.mortgage.airb_calculator import AIRBMortgageCalculator
from irbstudio.engine.mortgage.sa_calculator import SAMortgageCalculator

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

def run_simulation_with_memory_tracking(portfolio_size, 
                                       memory_efficient=True, 
                                       store_full_portfolio=False,
                                       n_iterations=5):
    """
    Run a simulation with memory tracking
    
    Args:
        portfolio_size: Number of loans in the portfolio
        memory_efficient: Whether to use memory-efficient mode
        store_full_portfolio: Whether to store full portfolio in results
        n_iterations: Number of Monte Carlo iterations
    
    Returns:
        Dict with memory usage statistics
    """
    # Create a synthetic portfolio with historical and recent data
    print(f"Creating synthetic portfolio with {portfolio_size} loans across multiple dates...")
    portfolio_data = []
    
    # Create reporting dates (6 months of historical data + 1 month of application data)
    reporting_dates = [
        '2025-04-30', '2025-05-31', '2025-06-30', 
        '2025-07-31', '2025-08-31', '2025-09-30', '2025-10-31'
    ]
    
    # Generate loans across multiple dates
    # We'll generate enough defaulted loans to ensure beta mixture fitting works
    num_default_loans = max(int(portfolio_size * 0.05), 100)  # Ensure at least 5% or 100 defaulting loans
    num_normal_loans = portfolio_size - num_default_loans
    
    # Create non-defaulting loans
    for i in range(num_normal_loans):
        loan_id = f'LOAN-{i}'
        
        # Create loan observations across time
        for month_idx, date in enumerate(reporting_dates):
            # Generate slightly different values per month
            pd_value = np.random.beta(1.5, 20) * 0.1  # Beta distribution for more realistic PD values
            score = np.random.uniform(1.5, 5.0)  # Direct score rather than log transformation
            exposure = np.random.uniform(100000, 500000)
            
            # Assign rating based on score
            if score > 4.0:
                rating = 'AAA'
            elif score > 3.5:
                rating = 'AA'
            elif score > 3.0:
                rating = 'A'
            elif score > 2.5:
                rating = 'BBB'
            else:
                rating = 'BB'
            
            # Add property value for SA calculator
            property_value = exposure / np.random.uniform(0.5, 0.9)  # LTV between 50% and 90%
            
            portfolio_data.append({
                'loan_id': loan_id,
                'reporting_date': date,
                'pd': pd_value,
                'lgd': 0.25,
                'exposure': exposure,
                'rating': rating,
                'segment': 'Mortgage',  # All mortgage for simplicity
                'score': score,
                'date': pd.to_datetime(date),
                'is_default': 0,
                'into_default': 0,
                'property_value': property_value,
                'ltv': exposure / property_value
            })
    
    # Create defaulting loans
    for i in range(num_default_loans):
        loan_id = f'LOAN-DEF-{i}'
        
        # Pick a month when default occurs (in the historical period)
        default_month = np.random.choice(range(1, 5))  # Default in months 1-4 
        
        # Create loan observations across time
        for month_idx, date in enumerate(reporting_dates):
            # Determine default status
            current_default = 1 if month_idx >= default_month else 0
            current_into_default = 1 if month_idx == default_month else 0
            
            # Generate score that worsens approaching default
            if month_idx < default_month:
                # Score decreases as we approach default
                score_base = np.random.uniform(1.0, 2.5)  # Lower scores for loans that will default
                score = score_base * (1.0 - 0.15 * month_idx)  # Decrease score as we approach default
                pd_value = np.random.beta(3, 3) * 0.3  # Higher PDs for loans that will default
            else:
                # After default
                score = np.random.uniform(0.1, 0.8)  # Very low scores for defaulted loans
                pd_value = 1.0  # PD of 1 for defaulted loans
            
            # Assign rating based on score
            if score > 4.0:
                rating = 'AAA'
            elif score > 3.5:
                rating = 'AA'
            elif score > 3.0:
                rating = 'A'
            elif score > 2.5:
                rating = 'BBB'
            elif score > 0.8:
                rating = 'BB'
            else:
                rating = 'CCC'  # Lowest rating for near-default or defaulted loans
            
            # After default, assign default rating
            if current_default == 1:
                rating = 'DEFAULT'
            
            exposure = np.random.uniform(100000, 500000)
            # Add property value for SA calculator - higher LTVs for defaulting loans
            property_value = exposure / np.random.uniform(0.8, 1.1)  # LTV between 80% and 110%
            
            portfolio_data.append({
                'loan_id': loan_id,
                'reporting_date': date,
                'pd': pd_value,
                'lgd': 0.35,  # Higher LGD for defaulting loans
                'exposure': exposure,
                'rating': rating,
                'segment': 'Mortgage',
                'score': score,
                'date': pd.to_datetime(date),
                'is_default': current_default,
                'into_default': current_into_default,
                'property_value': property_value,
                'ltv': exposure / property_value
            })
    
    portfolio_df = pd.DataFrame(portfolio_data)
    print(f"Portfolio created with shape: {portfolio_df.shape}")
    
    # Make sure all required columns are properly formatted
    portfolio_df['date'] = pd.to_datetime(portfolio_df['reporting_date'])
    
    # Create score to rating bounds (required by PortfolioSimulator)
    # Map ratings to score ranges
    score_to_rating_bounds = {
        'AAA': (4.0, float('inf')),
        'AA': (3.5, 4.0),
        'A': (3.0, 3.5),
        'BBB': (2.5, 3.0),
        'BB': (0.0, 2.5)
    }
    
    # Set up simulators and calculators
    simulator = PortfolioSimulator(
        portfolio_df=portfolio_df,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='date',
        default_col='is_default',
        into_default_flag_col='into_default',
        score_col='score',
        asset_correlation=0.15,  # Add this parameter to control PD shock correlation
        application_start_date=pd.to_datetime('2025-10-31')  # Last month is application period
    )
    
    # Prepare the simulator (required step)
    simulator.prepare_simulation()
    
    airb_calculator = AIRBMortgageCalculator({
        'asset_correlation': 0.15,
        'confidence_level': 0.999,
        'lgd': 0.25
    })
    
    sa_calculator = SAMortgageCalculator({
        'risk_weight_schedule': {
            'AAA': 0.35,
            'AA': 0.35,
            'A': 0.35,
            'BBB': 0.5,
            'BB': 0.75,
            'B': 1.0,
            'CCC': 1.5,
            'DEFAULT': 1.5
        }
    })
    
    # Create IntegratedAnalysis instance
    analyzer = IntegratedAnalysis()
    analyzer.add_calculator('airb', airb_calculator)
    analyzer.add_calculator('sa', sa_calculator)
    analyzer.add_scenario('baseline', simulator, n_iterations=n_iterations)
    
    # Track memory usage
    memory_stats = {
        'before_run': get_memory_usage(),
        'peak': 0,
        'after_run': 0,
        'after_gc': 0
    }
    
    # Run scenario with specified memory options
    print(f"Running scenario with memory_efficient={memory_efficient}, store_full_portfolio={store_full_portfolio}...")
    analyzer.run_scenario(
        'baseline',
        calculator_names=['airb', 'sa'],
        memory_efficient=memory_efficient,
        store_full_portfolio=store_full_portfolio
    )
    
    memory_stats['after_run'] = get_memory_usage()
    
    # Force garbage collection
    print("Running garbage collection...")
    gc.collect()
    
    memory_stats['after_gc'] = get_memory_usage()
    memory_stats['peak'] = memory_stats['after_run']  # Simplified peak tracking
    
    print(f"Memory usage (MB):")
    print(f"  Before run: {memory_stats['before_run']:.2f}")
    print(f"  Peak: {memory_stats['peak']:.2f}")
    print(f"  After run: {memory_stats['after_run']:.2f}")
    print(f"  After GC: {memory_stats['after_gc']:.2f}")
    print(f"  Memory increase: {memory_stats['after_gc'] - memory_stats['before_run']:.2f}")
    
    return memory_stats

def compare_memory_usage():
    """
    Compare memory usage across different options
    """
    # Small portfolio for demonstration - using 10,000 loans should be enough to show memory differences
    portfolio_size = 10000
    iterations = 5
    
    print("=" * 80)
    print("MEMORY USAGE COMPARISON")
    print("=" * 80)
    
    # Test 1: Full memory usage (no optimizations)
    print("\nTest 1: No memory optimization")
    stats1 = run_simulation_with_memory_tracking(
        portfolio_size=portfolio_size,
        memory_efficient=False,
        store_full_portfolio=True,
        n_iterations=iterations
    )
    
    # Test 2: Memory-efficient processing only
    print("\nTest 2: Memory-efficient processing only")
    stats2 = run_simulation_with_memory_tracking(
        portfolio_size=portfolio_size,
        memory_efficient=True,
        store_full_portfolio=True,
        n_iterations=iterations
    )
    
    # Test 3: Memory-efficient storage only
    print("\nTest 3: Memory-efficient storage only")
    stats3 = run_simulation_with_memory_tracking(
        portfolio_size=portfolio_size,
        memory_efficient=False,
        store_full_portfolio=False,
        n_iterations=iterations
    )
    
    # Test 4: Full memory optimization
    print("\nTest 4: Full memory optimization")
    stats4 = run_simulation_with_memory_tracking(
        portfolio_size=portfolio_size,
        memory_efficient=True,
        store_full_portfolio=False,
        n_iterations=iterations
    )
    
    # Create comparison chart
    labels = ['No Optimization', 'Efficient Processing', 'Efficient Storage', 'Full Optimization']
    peak_memory = [stats1['peak'], stats2['peak'], stats3['peak'], stats4['peak']]
    final_memory = [stats1['after_gc'], stats2['after_gc'], stats3['after_gc'], stats4['after_gc']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, peak_memory, width, label='Peak Memory')
    ax.bar(x + width/2, final_memory, width, label='Final Memory')
    
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate memory savings
    savings = {
        'efficient_processing': 100 * (1 - stats2['peak'] / stats1['peak']),
        'efficient_storage': 100 * (1 - stats3['peak'] / stats1['peak']),
        'full_optimization': 100 * (1 - stats4['peak'] / stats1['peak']),
    }
    
    print("\nMemory Savings:")
    print(f"Efficient processing only: {savings['efficient_processing']:.1f}%")
    print(f"Efficient storage only: {savings['efficient_storage']:.1f}%")
    print(f"Full optimization: {savings['full_optimization']:.1f}%")
    
    return {
        'test1': stats1,
        'test2': stats2,
        'test3': stats3,
        'test4': stats4,
        'savings': savings
    }

if __name__ == "__main__":
    compare_memory_usage()