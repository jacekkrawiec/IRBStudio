"""
Reporting and visualization module for IRBStudio.

This module provides functions for creating interactive visualizations
and HTML dashboards to present analysis results.
"""

from .dashboard import (
    create_rwa_distribution_plot,
    create_scenario_comparison_plot,
    create_waterfall_chart,
    create_summary_table,
    create_percentile_plot,
    create_rwa_by_date_plot,
    create_rwa_distribution_by_date_plot,
    generate_html_report
)

__all__ = [
    'create_rwa_distribution_plot',
    'create_scenario_comparison_plot',
    'create_waterfall_chart',
    'create_summary_table',
    'create_percentile_plot',
    'create_rwa_by_date_plot',
    'create_rwa_distribution_by_date_plot',
    'generate_html_report'
]
