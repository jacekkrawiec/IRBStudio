"""
Freddie Mac Date-Based RWA Analysis
====================================

Streamlined example focusing on:
1. Two scenario comparison (Baseline vs Improved Model)
2. RWA evolution over time with confidence intervals
3. Scenario differences across reporting dates
4. Statistical analysis of temporal RWA patterns

This is optimized for performance and focuses specifically on date-based insights.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.engine.mortgage.airb_calculator import AIRBMortgageCalculator


def create_rwa_comparison_by_date(results_baseline, results_improved):
    """
    Create a chart comparing RWA across dates for two scenarios with confidence intervals.
    
    Args:
        results_baseline: List of RWAResult objects for baseline scenario
        results_improved: List of RWAResult objects for improved scenario
    
    Returns:
        Plotly figure with confidence intervals
    """
    # Extract date-based RWA for baseline
    baseline_data = {}
    for result in results_baseline:
        if result.by_date:
            for date_str, date_data in result.by_date.items():
                if date_str not in baseline_data:
                    baseline_data[date_str] = []
                baseline_data[date_str].append(date_data['total_rwa'])
    
    # Extract date-based RWA for improved
    improved_data = {}
    for result in results_improved:
        if result.by_date:
            for date_str, date_data in result.by_date.items():
                if date_str not in improved_data:
                    improved_data[date_str] = []
                improved_data[date_str].append(date_data['total_rwa'])
    
    # Calculate statistics for each date
    dates = sorted(baseline_data.keys())
    
    baseline_mean = [np.mean(baseline_data[d]) for d in dates]
    baseline_p5 = [np.percentile(baseline_data[d], 5) for d in dates]
    baseline_p95 = [np.percentile(baseline_data[d], 95) for d in dates]
    
    improved_mean = [np.mean(improved_data[d]) for d in dates]
    improved_p5 = [np.percentile(improved_data[d], 5) for d in dates]
    improved_p95 = [np.percentile(improved_data[d], 95) for d in dates]
    
    # Calculate difference
    difference_mean = [improved_mean[i] - baseline_mean[i] for i in range(len(dates))]
    difference_pct = [(improved_mean[i] - baseline_mean[i]) / baseline_mean[i] * 100 
                      for i in range(len(dates))]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'RWA Over Time: Baseline vs Improved Model (with 90% Confidence Intervals)',
            'RWA Difference: Improved - Baseline (Capital Savings)'
        ),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Top plot: RWA comparison with confidence intervals
    # Baseline
    fig.add_trace(
        go.Scatter(
            x=dates, y=baseline_mean,
            name='Baseline (Mean)',
            line=dict(color='blue', width=2),
            mode='lines'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates, y=baseline_p95,
            name='Baseline (95th %ile)',
            line=dict(color='blue', width=0, dash='dot'),
            mode='lines',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates, y=baseline_p5,
            name='Baseline (90% CI)',
            line=dict(color='blue', width=0),
            mode='lines',
            fillcolor='rgba(0, 0, 255, 0.15)',
            fill='tonexty',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Improved
    fig.add_trace(
        go.Scatter(
            x=dates, y=improved_mean,
            name='Improved (Mean)',
            line=dict(color='green', width=2),
            mode='lines'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates, y=improved_p95,
            name='Improved (95th %ile)',
            line=dict(color='green', width=0, dash='dot'),
            mode='lines',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates, y=improved_p5,
            name='Improved (90% CI)',
            line=dict(color='green', width=0),
            mode='lines',
            fillcolor='rgba(0, 255, 0, 0.15)',
            fill='tonexty',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Bottom plot: Difference
    fig.add_trace(
        go.Scatter(
            x=dates, y=difference_mean,
            name='RWA Reduction',
            line=dict(color='darkgreen', width=2),
            mode='lines+markers',
            marker=dict(size=6),
            hovertemplate='Date: %{x}<br>RWA Reduction: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Reporting Date", row=2, col=1)
    fig.update_yaxes(title_text="RWA ($)", row=1, col=1, tickformat='$,.0s')
    fig.update_yaxes(title_text="RWA Difference ($)", row=2, col=1, tickformat='$,.0s')
    
    fig.update_layout(
        height=900,
        title_text="Scenario Comparison: RWA Evolution with Confidence Intervals",
        title_font_size=16,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_capital_savings_analysis(results_baseline, results_improved):
    """
    Create detailed analysis of capital savings from model improvement.
    """
    # Extract total RWA for each iteration
    baseline_rwa = [r.total_rwa for r in results_baseline]
    improved_rwa = [r.total_rwa for r in results_improved]
    
    # Calculate statistics
    baseline_mean = np.mean(baseline_rwa)
    improved_mean = np.mean(improved_rwa)
    savings_mean = baseline_mean - improved_mean
    savings_pct = (savings_mean / baseline_mean) * 100
    
    baseline_median = np.median(baseline_rwa)
    improved_median = np.median(improved_rwa)
    savings_median = baseline_median - improved_median
    
    # Get date-based analysis for most recent date
    baseline_last_date_rwa = []
    improved_last_date_rwa = []
    
    for result in results_baseline:
        if result.by_date:
            last_date = max(result.by_date.keys())
            baseline_last_date_rwa.append(result.by_date[last_date]['total_rwa'])
    
    for result in results_improved:
        if result.by_date:
            last_date = max(result.by_date.keys())
            improved_last_date_rwa.append(result.by_date[last_date]['total_rwa'])
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Total RWA Distribution (All Dates)',
            'Last Date RWA Distribution',
            'Capital Savings Distribution',
            'Percentile Comparison'
        ),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # Plot 1: Total RWA distributions
    fig.add_trace(
        go.Histogram(
            x=baseline_rwa,
            name='Baseline',
            opacity=0.7,
            marker_color='blue',
            nbinsx=20
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=improved_rwa,
            name='Improved',
            opacity=0.7,
            marker_color='green',
            nbinsx=20
        ),
        row=1, col=1
    )
    
    # Plot 2: Last date RWA distributions
    if baseline_last_date_rwa and improved_last_date_rwa:
        fig.add_trace(
            go.Histogram(
                x=baseline_last_date_rwa,
                name='Baseline (Last Date)',
                opacity=0.7,
                marker_color='blue',
                nbinsx=20,
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=improved_last_date_rwa,
                name='Improved (Last Date)',
                opacity=0.7,
                marker_color='green',
                nbinsx=20,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Plot 3: Savings distribution
    savings_dist = [baseline_rwa[i] - improved_rwa[i] for i in range(len(baseline_rwa))]
    fig.add_trace(
        go.Histogram(
            x=savings_dist,
            name='Capital Savings',
            marker_color='darkgreen',
            nbinsx=20,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Plot 4: Percentile comparison
    percentiles = [5, 25, 50, 75, 95]
    baseline_pct = [np.percentile(baseline_rwa, p) for p in percentiles]
    improved_pct = [np.percentile(improved_rwa, p) for p in percentiles]
    
    fig.add_trace(
        go.Bar(
            x=[f'P{p}' for p in percentiles],
            y=baseline_pct,
            name='Baseline',
            marker_color='blue',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=[f'P{p}' for p in percentiles],
            y=improved_pct,
            name='Improved',
            marker_color='green',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="RWA ($)", row=1, col=1, tickformat='$,.0s')
    fig.update_xaxes(title_text="RWA ($)", row=1, col=2, tickformat='$,.0s')
    fig.update_xaxes(title_text="Savings ($)", row=2, col=1, tickformat='$,.0s')
    fig.update_xaxes(title_text="Percentile", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="RWA ($)", row=2, col=2, tickformat='$,.0s')
    
    fig.update_layout(
        height=800,
        title_text=f"Capital Impact Analysis: Mean Savings of ${savings_mean:,.0f} ({savings_pct:.1f}%)",
        title_font_size=16,
        barmode='group',
        template='plotly_white'
    )
    
    return fig


def main():
    try:
        print("=" * 80)
        print("FREDDIE MAC DATE-BASED RWA ANALYSIS")
        print("=" * 80)
        
        # Step 1: Load data
        print("\n1. Loading Freddie Mac mortgage data...")
        data_path = '../data/sample_portfolio_data_fm.csv'
        
        # Define columns
        rating_col = 'rating'
        loan_id_col = 'Loan_Sequence_Number'
        date_col = 'reporting_date'
        default_col = 'default_flag'
        into_default_flag_col = 'into_default_flag'
        score_col = 'score'
        ltv_column = 'Estimated_Loan-to-Value_(ELTV)'
        exposure_col = 'Current_Actual_UPB'
        
        cols = [rating_col, loan_id_col, date_col, default_col, 
                into_default_flag_col, score_col, ltv_column, exposure_col]
        
        portfolio_df = pd.read_csv(data_path, usecols=cols, parse_dates=[date_col])
        
        print(f"   ✓ Loaded {portfolio_df.shape[0]:,} records")
        print(f"   ✓ Date range: {portfolio_df[date_col].min()} to {portfolio_df[date_col].max()}")
        print(f"   ✓ Unique dates: {portfolio_df[date_col].nunique()}")
        print(f"   ✓ Unique loans: {portfolio_df[loan_id_col].nunique():,}")
        
        # Step 2: Define rating boundaries
        print("\n2. Setting up score-to-rating boundaries...")
        score_to_rating_bounds = {
            '1': (-1, 0.003613294451497495),
            '2': (0.003613294451497495, 0.005780360195785761),
            '3': (0.005780360195785761, 0.03225071728229523),
            '4': (0.03225071728229523, 0.039578670635819435),
            '5': (0.039578670635819435, 0.256146103143692),
            '6': (0.256146103143692, 0.7653337121009827),
            '7': (0.7653337121009827, 50)
        }
        print("   ✓ 7 rating grades defined")
        
        # Step 3: Configure AIRB calculator
        print("\n3. Configuring AIRB calculator...")
        airb_params = {
            'asset_correlation': 0.15,
            'confidence_level': 0.999,
            'lgd': 0.25,
            'maturity_adjustment': False
        }
        airb_calculator = AIRBMortgageCalculator(airb_params)
        print("   ✓ AIRB calculator configured (ρ=0.15, LGD=0.25)")
        
        # Step 4: Initialize analysis framework
        print("\n4. Initializing IntegratedAnalysis...")
        analysis = IntegratedAnalysis(date_column=date_col)
        analysis.add_calculator('AIRB', airb_calculator)
        print("   ✓ Framework initialized with date breakdown enabled")
        
        # Step 5: Create scenarios
        print("\n5. Creating scenarios...")
        application_start_date = datetime(2024, 4, 1)
        n_iterations = 15
        
        # Baseline: Current model performance
        baseline_simulator = PortfolioSimulator(
            portfolio_df=portfolio_df.copy(),
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col=rating_col,
            loan_id_col=loan_id_col,
            date_col=date_col,
            default_col=default_col,
            into_default_flag_col=into_default_flag_col,
            score_col=score_col,
            exposure_col=exposure_col,
            application_start_date=application_start_date,
            asset_correlation=0.15,
            target_auc=0.80,
            random_seed=42
        )
        baseline_simulator.prepare_simulation()
        analysis.add_scenario('Baseline', baseline_simulator, n_iterations=n_iterations)
        print(f"   ✓ Baseline (AUC=0.80, {n_iterations} iterations)")
        
        # Improved: Enhanced model performance
        improved_simulator = PortfolioSimulator(
            portfolio_df=portfolio_df.copy(),
            score_to_rating_bounds=score_to_rating_bounds,
            rating_col=rating_col,
            loan_id_col=loan_id_col,
            date_col=date_col,
            default_col=default_col,
            into_default_flag_col=into_default_flag_col,
            score_col=score_col,
            exposure_col=exposure_col,
            application_start_date=application_start_date,
            asset_correlation=0.15,
            target_auc=0.90,
            random_seed=42
        )
        improved_simulator.prepare_simulation()
        analysis.add_scenario('Improved', improved_simulator, n_iterations=n_iterations)
        print(f"   ✓ Improved (AUC=0.90, {n_iterations} iterations)")
        
        # Step 6: Run simulations
        print(f"\n6. Running Monte Carlo simulations...")
        print(f"   Total: 2 scenarios × {n_iterations} iterations = {2 * n_iterations} simulations")
        print("   (This will take a few minutes with date breakdown enabled)")
        
        for scenario_name in ['Baseline', 'Improved']:
            print(f"\n   Running: {scenario_name}...")
            analysis.run_scenario(
                scenario_name=scenario_name,
                calculator_names=['AIRB'],
                memory_efficient=True,
                store_full_portfolio=False,
                process_all_dates=True
            )
        
        print(f"\n   ✓ Completed all simulations")
        
        # Step 7: Extract results
        print("\n7. Extracting results...")
        baseline_results = analysis.results['Baseline']['calculator_results']['AIRB']['results']
        improved_results = analysis.results['Improved']['calculator_results']['AIRB']['results']
        
        print(f"   ✓ Baseline: {len(baseline_results)} iterations")
        print(f"   ✓ Improved: {len(improved_results)} iterations")
        
        # Verify date breakdown
        if baseline_results[0].by_date:
            n_dates = len(baseline_results[0].by_date)
            print(f"   ✓ Date breakdown: {n_dates} unique dates")
        else:
            print("   ⚠ Warning: No date breakdown found!")
        
        # Step 8: Create visualizations
        print("\n8. Creating visualizations...")
        output_dir = 'freddie_mac_date_analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # Chart 1: RWA comparison with confidence intervals
        print("   - RWA comparison over time with confidence intervals...")
        fig1 = create_rwa_comparison_by_date(baseline_results, improved_results)
        fig1.write_html(os.path.join(output_dir, 'rwa_comparison_ci.html'))
        print("     ✓ Saved: rwa_comparison_ci.html")
        
        # Chart 2: Capital savings analysis
        print("   - Capital savings analysis...")
        fig2 = create_capital_savings_analysis(baseline_results, improved_results)
        fig2.write_html(os.path.join(output_dir, 'capital_savings.html'))
        print("     ✓ Saved: capital_savings.html")
        
        # Step 9: Display summary statistics
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        
        baseline_rwa = [r.total_rwa for r in baseline_results]
        improved_rwa = [r.total_rwa for r in improved_results]
        
        print("\nBaseline Scenario (AUC=0.80):")
        print(f"  Mean RWA:    ${np.mean(baseline_rwa):,.0f}")
        print(f"  Median RWA:  ${np.median(baseline_rwa):,.0f}")
        print(f"  Std Dev:     ${np.std(baseline_rwa):,.0f}")
        print(f"  P5:          ${np.percentile(baseline_rwa, 5):,.0f}")
        print(f"  P95:         ${np.percentile(baseline_rwa, 95):,.0f}")
        
        print("\nImproved Scenario (AUC=0.90):")
        print(f"  Mean RWA:    ${np.mean(improved_rwa):,.0f}")
        print(f"  Median RWA:  ${np.median(improved_rwa):,.0f}")
        print(f"  Std Dev:     ${np.std(improved_rwa):,.0f}")
        print(f"  P5:          ${np.percentile(improved_rwa, 5):,.0f}")
        print(f"  P95:         ${np.percentile(improved_rwa, 95):,.0f}")
        
        savings_mean = np.mean(baseline_rwa) - np.mean(improved_rwa)
        savings_pct = (savings_mean / np.mean(baseline_rwa)) * 100
        capital_savings = savings_mean * 0.08  # 8% capital requirement
        
        print("\nCapital Impact:")
        print(f"  Mean RWA Reduction:      ${savings_mean:,.0f}")
        print(f"  Reduction (%):           {savings_pct:.2f}%")
        print(f"  Capital Savings (8%):    ${capital_savings:,.0f}")
        
        print("\n" + "=" * 80)
        print("VISUALIZATIONS CREATED")
        print("=" * 80)
        print(f"\nOutput directory: {output_dir}/")
        print("\n📊 Generated Charts:")
        print("  1. rwa_comparison_ci.html     - RWA over time with 90% confidence intervals")
        print("  2. capital_savings.html       - Detailed capital impact analysis")
        print("\n💡 Key Insights:")
        print("  • Time series shows RWA evolution across all reporting dates")
        print("  • Confidence intervals (P5-P95) show simulation uncertainty")
        print("  • Lower panel shows capital savings from model improvement")
        print("  • All charts are interactive (hover, zoom, pan)")
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
