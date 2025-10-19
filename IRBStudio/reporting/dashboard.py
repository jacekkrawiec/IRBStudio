"""
Dashboard generation module for IRBStudio.

This module provides functions to create interactive Plotly visualizations
and comprehensive HTML reports for portfolio analysis results.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_rwa_distribution_plot(
    results: Dict[str, Dict[str, List[float]]],
    scenario_name: str,
    calculator_name: str = 'AIRB',
    title: Optional[str] = None,
    show_stats: bool = True
) -> go.Figure:
    """
    Create a distribution plot showing RWA values across Monte Carlo iterations.
    
    Args:
        results: Dictionary containing scenario results with RWA values
        scenario_name: Name of the scenario to plot
        calculator_name: Name of the calculator ('AIRB' or 'SA')
        title: Custom title for the plot (auto-generated if None)
        show_stats: Whether to show statistical annotations
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> results = {'Baseline': {'AIRB': {'rwa_values': [100, 102, 98, ...]}}}
        >>> fig = create_rwa_distribution_plot(results, 'Baseline', 'AIRB')
        >>> fig.show()
    """
    logger.info(f"Creating RWA distribution plot for scenario '{scenario_name}' with calculator '{calculator_name}'")
    
    # Extract RWA values
    try:
        scenario_data = results.get(scenario_name, {})
        calc_data = scenario_data.get(calculator_name, {})
        
        # Handle different data structures
        if isinstance(calc_data, dict) and 'rwa_values' in calc_data:
            rwa_values = calc_data['rwa_values']
        elif isinstance(calc_data, list):
            rwa_values = calc_data
        else:
            raise ValueError(f"Unable to extract RWA values for {scenario_name}/{calculator_name}")
        
    except Exception as e:
        logger.error(f"Error extracting RWA values: {e}")
        raise
    
    # Calculate statistics
    mean_rwa = np.mean(rwa_values)
    median_rwa = np.median(rwa_values)
    std_rwa = np.std(rwa_values)
    p5 = np.percentile(rwa_values, 5)
    p95 = np.percentile(rwa_values, 95)
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=rwa_values,
        name='RWA Distribution',
        marker=dict(
            color='rgba(55, 128, 191, 0.7)',
            line=dict(color='rgba(55, 128, 191, 1.0)', width=1)
        ),
        nbinsx=30,
        hovertemplate='RWA: %{x:,.0f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add statistical markers if requested
    if show_stats:
        # Mean line
        fig.add_vline(
            x=mean_rwa,
            line=dict(color='red', width=2, dash='dash'),
            annotation_text=f'Mean: ${mean_rwa:,.0f}',
            annotation_position='top'
        )
        
        # Median line
        fig.add_vline(
            x=median_rwa,
            line=dict(color='green', width=2, dash='dash'),
            annotation_text=f'Median: ${median_rwa:,.0f}',
            annotation_position='bottom'
        )
        
        # 90% confidence interval
        fig.add_vrect(
            x0=p5, x1=p95,
            fillcolor='rgba(255, 200, 0, 0.1)',
            layer='below',
            line_width=0,
            annotation_text='90% CI',
            annotation_position='top right'
        )
    
    # Update layout
    if title is None:
        title = f'RWA Distribution: {scenario_name} ({calculator_name})'
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        xaxis_title='Risk-Weighted Assets ($)',
        yaxis_title='Frequency',
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    # Format x-axis as currency
    fig.update_xaxes(tickformat='$,.0f')
    
    logger.info(f"Distribution plot created successfully: mean=${mean_rwa:,.0f}, std=${std_rwa:,.0f}")
    
    return fig


def create_scenario_comparison_plot(
    results: Dict[str, Dict[str, Dict[str, float]]],
    calculator_name: str = 'AIRB',
    baseline_scenario: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a bar chart comparing mean RWA across different scenarios.
    
    Args:
        results: Dictionary containing scenario results
        calculator_name: Name of the calculator to compare
        baseline_scenario: Name of baseline scenario for delta calculations
        title: Custom title for the plot
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> results = {
        ...     'Baseline': {'AIRB': {'mean': 100000, 'std': 5000}},
        ...     'Optimistic': {'AIRB': {'mean': 95000, 'std': 4800}}
        ... }
        >>> fig = create_scenario_comparison_plot(results, 'AIRB', 'Baseline')
        >>> fig.show()
    """
    logger.info(f"Creating scenario comparison plot for calculator '{calculator_name}'")
    
    # Extract data for plotting
    scenarios = []
    mean_values = []
    std_values = []
    
    for scenario_name, scenario_data in results.items():
        calc_data = scenario_data.get(calculator_name, {})
        if calc_data:
            scenarios.append(scenario_name)
            mean_values.append(calc_data.get('mean', 0))
            std_values.append(calc_data.get('std', 0))
    
    if not scenarios:
        raise ValueError(f"No data found for calculator '{calculator_name}'")
    
    # Calculate deltas if baseline is specified
    deltas = []
    if baseline_scenario and baseline_scenario in scenarios:
        baseline_idx = scenarios.index(baseline_scenario)
        baseline_mean = mean_values[baseline_idx]
        deltas = [(val - baseline_mean) for val in mean_values]
    
    # Create figure
    fig = go.Figure()
    
    # Add mean RWA bars
    fig.add_trace(go.Bar(
        x=scenarios,
        y=mean_values,
        name='Mean RWA',
        marker=dict(
            color=['rgba(55, 128, 191, 0.8)' if s != baseline_scenario else 'rgba(219, 64, 82, 0.8)' 
                   for s in scenarios],
            line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
        ),
        error_y=dict(
            type='data',
            array=std_values,
            visible=True,
            color='rgba(0, 0, 0, 0.3)'
        ),
        text=[f'${val:,.0f}' for val in mean_values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Mean RWA: %{y:$,.0f}<br>Std Dev: %{error_y.array:$,.0f}<extra></extra>'
    ))
    
    # Add delta annotations if applicable
    if deltas and baseline_scenario:
        for i, (scenario, delta) in enumerate(zip(scenarios, deltas)):
            if scenario != baseline_scenario:
                pct_change = (delta / baseline_mean) * 100
                color = 'green' if delta < 0 else 'red'
                fig.add_annotation(
                    x=scenario,
                    y=mean_values[i] + std_values[i],
                    text=f'{delta:+,.0f} ({pct_change:+.2f}%)',
                    showarrow=False,
                    yshift=20,
                    font=dict(color=color, size=11)
                )
    
    # Update layout
    if title is None:
        title = f'Scenario Comparison: Mean RWA ({calculator_name})'
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        xaxis_title='Scenario',
        yaxis_title='Mean Risk-Weighted Assets ($)',
        showlegend=False,
        template='plotly_white',
        height=500,
        hovermode='x'
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickformat='$,.0f')
    
    logger.info(f"Scenario comparison plot created with {len(scenarios)} scenarios")
    
    return fig


def create_waterfall_chart(
    baseline_scenario: str,
    comparison_scenario: str,
    results: Dict[str, Dict[str, Dict[str, float]]],
    calculator_name: str = 'AIRB',
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a waterfall chart showing the transition from baseline to comparison scenario.
    
    Args:
        baseline_scenario: Name of the baseline scenario
        comparison_scenario: Name of the comparison scenario
        results: Dictionary containing scenario results
        calculator_name: Name of the calculator
        title: Custom title for the plot
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> fig = create_waterfall_chart('Baseline', 'Improved Model', results, 'AIRB')
        >>> fig.show()
    """
    logger.info(f"Creating waterfall chart: {baseline_scenario} → {comparison_scenario}")
    
    # Extract baseline and comparison values
    baseline_mean = results[baseline_scenario][calculator_name]['mean']
    comparison_mean = results[comparison_scenario][calculator_name]['mean']
    delta = comparison_mean - baseline_mean
    
    # Create waterfall data
    x_data = [baseline_scenario, 'Change', comparison_scenario]
    y_data = [baseline_mean, delta, comparison_mean]
    
    # Determine measure types (absolute, relative, total)
    measure = ['absolute', 'relative', 'total']
    
    # Create figure
    fig = go.Figure(go.Waterfall(
        x=x_data,
        y=y_data,
        measure=measure,
        text=[f'${baseline_mean:,.0f}', f'{delta:+,.0f}', f'${comparison_mean:,.0f}'],
        textposition='outside',
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
        decreasing={'marker': {'color': 'rgba(76, 175, 80, 0.8)'}},
        increasing={'marker': {'color': 'rgba(244, 67, 54, 0.8)'}},
        totals={'marker': {'color': 'rgba(33, 150, 243, 0.8)'}},
        hovertemplate='%{x}<br>RWA: %{y:$,.0f}<extra></extra>'
    ))
    
    # Update layout
    if title is None:
        pct_change = (delta / baseline_mean) * 100
        title = f'RWA Impact: {baseline_scenario} → {comparison_scenario} ({pct_change:+.2f}%)'
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        yaxis_title='Risk-Weighted Assets ($)',
        showlegend=False,
        template='plotly_white',
        height=500
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickformat='$,.0f')
    
    logger.info(f"Waterfall chart created: ${delta:,.0f} change ({(delta/baseline_mean)*100:.2f}%)")
    
    return fig


def create_summary_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    calculator_names: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a summary table showing key statistics for all scenarios.
    
    Args:
        results: Dictionary containing scenario results
        calculator_names: List of calculators to include (None = all)
    
    Returns:
        Plotly Figure object containing a table
    
    Example:
        >>> fig = create_summary_table(results, ['AIRB', 'SA'])
        >>> fig.show()
    """
    logger.info("Creating summary statistics table")
    
    # Prepare table data
    scenarios = []
    calculators = []
    means = []
    medians = []
    stds = []
    mins = []
    maxs = []
    p5s = []
    p95s = []
    
    for scenario_name, scenario_data in results.items():
        for calc_name, calc_data in scenario_data.items():
            if calculator_names and calc_name not in calculator_names:
                continue
            
            if calc_data:
                scenarios.append(scenario_name)
                calculators.append(calc_name)
                means.append(f"${calc_data.get('mean', 0):,.0f}")
                medians.append(f"${calc_data.get('median', 0):,.0f}")
                stds.append(f"${calc_data.get('std', 0):,.0f}")
                mins.append(f"${calc_data.get('min', 0):,.0f}")
                maxs.append(f"${calc_data.get('max', 0):,.0f}")
                p5s.append(f"${calc_data.get('p5', 0):,.0f}")
                p95s.append(f"${calc_data.get('p95', 0):,.0f}")
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Scenario</b>', '<b>Calculator</b>', '<b>Mean</b>', '<b>Median</b>', 
                    '<b>Std Dev</b>', '<b>Min</b>', '<b>Max</b>', '<b>P5</b>', '<b>P95</b>'],
            fill_color='rgba(55, 128, 191, 0.8)',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[scenarios, calculators, means, medians, stds, mins, maxs, p5s, p95s],
            fill_color=[['rgba(245, 245, 245, 1)' if i % 2 == 0 else 'white' for i in range(len(scenarios))]],
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=dict(
            text='Summary Statistics',
            font=dict(size=20)
        ),
        height=max(300, len(scenarios) * 40 + 100),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    logger.info(f"Summary table created with {len(scenarios)} rows")
    
    return fig


def create_percentile_plot(
    results: Dict[str, Dict[str, List[float]]],
    scenario_name: str,
    calculator_name: str = 'AIRB',
    percentiles: List[int] = [5, 10, 25, 50, 75, 90, 95],
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a percentile visualization showing the distribution of RWA values.
    
    Args:
        results: Dictionary containing scenario results
        scenario_name: Name of the scenario
        calculator_name: Name of the calculator
        percentiles: List of percentiles to display
        title: Custom title for the plot
    
    Returns:
        Plotly Figure object
    """
    logger.info(f"Creating percentile plot for {scenario_name}/{calculator_name}")
    
    # Extract RWA values
    scenario_data = results.get(scenario_name, {})
    calc_data = scenario_data.get(calculator_name, {})
    
    if isinstance(calc_data, dict) and 'rwa_values' in calc_data:
        rwa_values = calc_data['rwa_values']
    elif isinstance(calc_data, list):
        rwa_values = calc_data
    else:
        raise ValueError(f"Unable to extract RWA values for {scenario_name}/{calculator_name}")
    
    # Calculate percentiles
    percentile_values = np.percentile(rwa_values, percentiles)
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=[f'P{p}' for p in percentiles],
        y=percentile_values,
        marker=dict(
            color=percentile_values,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='RWA ($)')
        ),
        text=[f'${val:,.0f}' for val in percentile_values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>RWA: %{y:$,.0f}<extra></extra>'
    ))
    
    # Update layout
    if title is None:
        title = f'RWA Percentiles: {scenario_name} ({calculator_name})'
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        xaxis_title='Percentile',
        yaxis_title='Risk-Weighted Assets ($)',
        showlegend=False,
        template='plotly_white',
        height=500
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickformat='$,.0f')
    
    logger.info(f"Percentile plot created successfully")
    
    return fig


def create_rwa_by_date_plot(
    results_by_iteration: List[Any],
    scenario_name: str,
    calculator_name: str = 'AIRB',
    title: Optional[str] = None,
    show_last_date_only: bool = False
) -> go.Figure:
    """
    Create a line plot showing RWA values by date across all simulation iterations.
    Each iteration is shown as a separate series/line.
    
    Args:
        results_by_iteration: List of result objects from IntegratedAnalysis, each with 'by_date' field
        scenario_name: Name of the scenario
        calculator_name: Name of the calculator ('AIRB' or 'SA')
        title: Custom title for the plot (auto-generated if None)
        show_last_date_only: If True, only show RWA for the last date
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> results = integrated_analysis.results['Baseline']['calculator_results']['AIRB']['results']
        >>> fig = create_rwa_by_date_plot(results, 'Baseline', 'AIRB')
        >>> fig.show()
    """
    logger.info(f"Creating RWA by date plot for scenario '{scenario_name}' with calculator '{calculator_name}'")
    
    fig = go.Figure()
    
    # Extract dates and RWA values from all iterations
    all_dates = set()
    iteration_data = []
    
    for i, result in enumerate(results_by_iteration):
        if hasattr(result, 'by_date') and result.by_date:
            dates = []
            rwa_values = []
            
            for date_str, date_data in sorted(result.by_date.items()):
                dates.append(pd.to_datetime(date_str))
                rwa_values.append(date_data['total_rwa'])
                all_dates.add(date_str)
            
            iteration_data.append({
                'iteration': i + 1,
                'dates': dates,
                'rwa_values': rwa_values
            })
    
    if not iteration_data:
        logger.warning("No date-based data found in results")
        # Return empty figure with message
        fig.add_annotation(
            text="No date-based RWA data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Filter to last date only if requested
    if show_last_date_only:
        last_date = max(pd.to_datetime(d) for d in all_dates)
        for data in iteration_data:
            if data['dates']:
                last_idx = data['dates'].index(max(data['dates']))
                data['dates'] = [data['dates'][last_idx]]
                data['rwa_values'] = [data['rwa_values'][last_idx]]
    
    # Add a line for each iteration
    for data in iteration_data:
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['rwa_values'],
            mode='lines+markers',
            name=f"Iteration {data['iteration']}",
            line=dict(width=1.5),
            marker=dict(size=6),
            hovertemplate='<b>Iteration %{fullData.name}</b><br>Date: %{x|%Y-%m-%d}<br>RWA: $%{y:,.0f}<extra></extra>'
        ))
    
    # Calculate mean RWA by date for reference line
    date_rwa_map = {}
    for data in iteration_data:
        for date, rwa in zip(data['dates'], data['rwa_values']):
            if date not in date_rwa_map:
                date_rwa_map[date] = []
            date_rwa_map[date].append(rwa)
    
    sorted_dates = sorted(date_rwa_map.keys())
    mean_rwa_values = [np.mean(date_rwa_map[date]) for date in sorted_dates]
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=sorted_dates,
        y=mean_rwa_values,
        mode='lines',
        name='Mean RWA',
        line=dict(color='red', width=3, dash='dash'),
        hovertemplate='<b>Mean RWA</b><br>Date: %{x|%Y-%m-%d}<br>RWA: $%{y:,.0f}<extra></extra>'
    ))
    
    # Update layout
    if title is None:
        date_suffix = " (Last Date Only)" if show_last_date_only else ""
        title = f'RWA by Date: {scenario_name} ({calculator_name}){date_suffix}'
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        xaxis_title='Reporting Date',
        yaxis_title='RWA ($)',
        template='plotly_white',
        height=600,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickformat='$,.0f')
    
    logger.info(f"RWA by date plot created successfully with {len(iteration_data)} iterations")
    
    return fig


def create_rwa_distribution_by_date_plot(
    results_by_iteration: List[Any],
    scenario_name: str,
    calculator_name: str = 'AIRB',
    specific_date: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a distribution plot showing RWA values for a specific date across all iterations.
    
    Args:
        results_by_iteration: List of result objects from IntegratedAnalysis
        scenario_name: Name of the scenario
        calculator_name: Name of the calculator ('AIRB' or 'SA')
        specific_date: Specific date to show (None = last date)
        title: Custom title for the plot (auto-generated if None)
    
    Returns:
        Plotly Figure object
    """
    logger.info(f"Creating RWA distribution by date plot for scenario '{scenario_name}' with calculator '{calculator_name}'")
    
    # Extract RWA values for the specific date
    rwa_values = []
    target_date = specific_date
    
    for result in results_by_iteration:
        if hasattr(result, 'by_date') and result.by_date:
            # If no specific date provided, use the last date
            if target_date is None:
                target_date = max(result.by_date.keys())
            
            if target_date in result.by_date:
                rwa_values.append(result.by_date[target_date]['total_rwa'])
    
    if not rwa_values:
        logger.warning(f"No RWA data found for date {target_date}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"No RWA data available for date {target_date}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Calculate statistics
    mean_rwa = np.mean(rwa_values)
    std_rwa = np.std(rwa_values)
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=rwa_values,
        nbinsx=20,
        name='RWA Distribution',
        marker=dict(
            color='steelblue',
            line=dict(color='white', width=1)
        ),
        hovertemplate='RWA Range: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    fig.add_vline(
        x=mean_rwa,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${mean_rwa:,.0f}",
        annotation_position="top"
    )
    
    # Update layout
    if title is None:
        title = f'RWA Distribution for {target_date}: {scenario_name} ({calculator_name})'
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        xaxis_title='RWA ($)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=500,
        showlegend=False,
        annotations=[
            dict(
                text=f'Mean: ${mean_rwa:,.0f}<br>Std Dev: ${std_rwa:,.0f}<br>N={len(rwa_values)}',
                xref='paper', yref='paper',
                x=0.98, y=0.98,
                xanchor='right', yanchor='top',
                showarrow=False,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=12)
            )
        ]
    )
    
    # Format x-axis as currency
    fig.update_xaxes(tickformat='$,.0f')
    
    logger.info(f"RWA distribution by date plot created successfully: mean=${mean_rwa:,.0f}, std=${std_rwa:,.0f}")
    
    return fig


def generate_html_report(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = 'IRBStudio Analysis Report',
    include_plots: Optional[List[str]] = None
) -> Path:
    """
    Generate a comprehensive HTML report with all visualizations.
    
    Args:
        results: Complete results dictionary from run_analysis()
        output_path: Path where HTML file should be saved
        title: Title for the report
        include_plots: List of plot types to include (None = all)
                      Options: 'distribution', 'comparison', 'waterfall', 'summary', 'percentile'
    
    Returns:
        Path to the generated HTML file
    
    Example:
        >>> results = run_analysis(config_path, portfolio_path)
        >>> report_path = generate_html_report(results, 'analysis_report.html')
        >>> print(f"Report saved to: {report_path}")
    """
    logger.info(f"Generating HTML report: {output_path}")
    
    output_path = Path(output_path)
    
    # Default to all plot types
    if include_plots is None:
        include_plots = ['distribution', 'comparison', 'waterfall', 'summary', 'percentile']
    
    # Extract data from results
    scenario_results = results.get('scenarios', {})
    comparisons = results.get('comparisons', {})
    portfolio_stats = results.get('portfolio_stats', {})
    execution_time = results.get('execution_time', 0)
    timestamp = results.get('analysis_timestamp', datetime.now().isoformat())
    
    # Start building HTML
    html_parts = []
    
    # HTML header
    html_parts.append(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #3780bf;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }}
        .metadata {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .metadata p {{
            margin: 5px 0;
            color: #666;
        }}
        .plot-container {{
            margin: 30px 0;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #999;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="metadata">
            <p><strong>Analysis Date:</strong> {timestamp}</p>
            <p><strong>Execution Time:</strong> {execution_time:.2f} seconds</p>
            <p><strong>Number of Scenarios:</strong> {len(scenario_results)}</p>
            <p><strong>Total Loans:</strong> {portfolio_stats.get('n_loans', 0):,}</p>
            <p><strong>Total Exposure:</strong> ${portfolio_stats.get('total_exposure', 0):,.2f}</p>
        </div>
""")
    
    # Get list of calculators used
    calculators = set()
    for scenario_data in scenario_results.values():
        calculators.update(scenario_data.keys())
    calculators = sorted(list(calculators))
    
    # Add summary table if requested
    if 'summary' in include_plots:
        html_parts.append("<h2>Summary Statistics</h2>")
        html_parts.append('<div class="plot-container" id="summary-table"></div>')
        
        try:
            fig = create_summary_table(scenario_results, calculators)
            html_parts.append(f"""
<script>
    var summaryData = {fig.to_json()};
    Plotly.newPlot('summary-table', summaryData.data, summaryData.layout);
</script>
""")
        except Exception as e:
            logger.warning(f"Could not create summary table: {e}")
    
    # Add comparison plot if requested
    if 'comparison' in include_plots and len(scenario_results) > 1:
        html_parts.append("<h2>Scenario Comparison</h2>")
        
        for calc_name in calculators:
            html_parts.append(f'<div class="plot-container" id="comparison-{calc_name}"></div>')
            
            try:
                baseline = list(scenario_results.keys())[0]
                fig = create_scenario_comparison_plot(scenario_results, calc_name, baseline)
                html_parts.append(f"""
<script>
    var comparisonData_{calc_name} = {fig.to_json()};
    Plotly.newPlot('comparison-{calc_name}', comparisonData_{calc_name}.data, comparisonData_{calc_name}.layout);
</script>
""")
            except Exception as e:
                logger.warning(f"Could not create comparison plot for {calc_name}: {e}")
    
    # Add distribution plots if requested
    if 'distribution' in include_plots:
        html_parts.append("<h2>RWA Distributions</h2>")
        
        for scenario_name in scenario_results.keys():
            for calc_name in calculators:
                plot_id = f"dist-{scenario_name.replace(' ', '_')}-{calc_name}"
                html_parts.append(f'<div class="plot-container" id="{plot_id}"></div>')
                
                try:
                    # Need to restructure data for this function
                    plot_data = {
                        scenario_name: {
                            calc_name: {
                                'rwa_values': [
                                    r.total_rwa for r in scenario_results[scenario_name].get(calc_name, {}).get('results', [])
                                    if hasattr(r, 'total_rwa')
                                ]
                            }
                        }
                    }
                    
                    if plot_data[scenario_name][calc_name]['rwa_values']:
                        fig = create_rwa_distribution_plot(plot_data, scenario_name, calc_name)
                        html_parts.append(f"""
<script>
    var distData_{scenario_name.replace(' ', '_')}_{calc_name} = {fig.to_json()};
    Plotly.newPlot('{plot_id}', distData_{scenario_name.replace(' ', '_')}_{calc_name}.data, distData_{scenario_name.replace(' ', '_')}_{calc_name}.layout);
</script>
""")
                except Exception as e:
                    logger.warning(f"Could not create distribution plot for {scenario_name}/{calc_name}: {e}")
    
    # Add waterfall charts if requested
    if 'waterfall' in include_plots and len(scenario_results) > 1:
        html_parts.append("<h2>Scenario Impact (Waterfall)</h2>")
        
        baseline = list(scenario_results.keys())[0]
        for scenario_name in list(scenario_results.keys())[1:]:
            for calc_name in calculators:
                plot_id = f"waterfall-{scenario_name.replace(' ', '_')}-{calc_name}"
                html_parts.append(f'<div class="plot-container" id="{plot_id}"></div>')
                
                try:
                    fig = create_waterfall_chart(baseline, scenario_name, scenario_results, calc_name)
                    html_parts.append(f"""
<script>
    var waterfallData_{scenario_name.replace(' ', '_')}_{calc_name} = {fig.to_json()};
    Plotly.newPlot('{plot_id}', waterfallData_{scenario_name.replace(' ', '_')}_{calc_name}.data, waterfallData_{scenario_name.replace(' ', '_')}_{calc_name}.layout);
</script>
""")
                except Exception as e:
                    logger.warning(f"Could not create waterfall chart for {baseline} → {scenario_name}/{calc_name}: {e}")
    
    # HTML footer
    html_parts.append(f"""
        <div class="footer">
            <p>Generated by IRBStudio v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
""")
    
    # Write to file
    html_content = '\n'.join(html_parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')
    
    logger.info(f"HTML report generated successfully: {output_path}")
    
    return output_path
