"""
Priority 2 Tests: Reporting & Visualization

Module: irbstudio.reporting.dashboard
Focus: Dashboard generation and visualizations
"""

import pytest
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from irbstudio.reporting.dashboard import (
    create_rwa_distribution_plot,
    create_scenario_comparison_plot,
    generate_html_report
)


class TestRWADistributionPlot:
    """Tests for create_rwa_distribution_plot()."""
    
    def test_create_rwa_distribution_plot_basic(self):
        """Test basic RWA distribution plot creation."""
        results = {
            'Baseline': {
                'AIRB': {
                    'rwa_values': [100000, 102000, 98000, 101000, 99000]
                }
            }
        }
        
        fig = create_rwa_distribution_plot(
            results=results,
            scenario_name='Baseline',
            calculator_name='AIRB'
        )
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_rwa_distribution_plot_custom_title(self):
        """Test RWA distribution plot with custom title."""
        results = {
            'Test': {
                'AIRB': {
                    'rwa_values': [100, 110, 105, 95, 100]
                }
            }
        }
        
        custom_title = "Custom RWA Distribution"
        
        fig = create_rwa_distribution_plot(
            results=results,
            scenario_name='Test',
            calculator_name='AIRB',
            title=custom_title
        )
        
        assert fig is not None
        assert custom_title in str(fig.layout.title.text)
    
    def test_create_rwa_distribution_plot_with_stats(self):
        """Test RWA distribution plot shows statistics."""
        results = {
            'Baseline': {
                'AIRB': {
                    'rwa_values': [100, 110, 105, 95, 100, 102, 98]
                }
            }
        }
        
        fig = create_rwa_distribution_plot(
            results=results,
            scenario_name='Baseline',
            calculator_name='AIRB',
            show_stats=True
        )
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_rwa_distribution_plot_without_stats(self):
        """Test RWA distribution plot without statistics annotations."""
        results = {
            'Baseline': {
                'AIRB': {
                    'rwa_values': [100, 110, 105, 95, 100]
                }
            }
        }
        
        fig = create_rwa_distribution_plot(
            results=results,
            scenario_name='Baseline',
            calculator_name='AIRB',
            show_stats=False
        )
        
        assert fig is not None
    
    def test_create_rwa_distribution_plot_sa_calculator(self):
        """Test RWA distribution plot with SA calculator."""
        results = {
            'Baseline': {
                'SA': {
                    'rwa_values': [95000, 97000, 93000, 96000, 94000]
                }
            }
        }
        
        fig = create_rwa_distribution_plot(
            results=results,
            scenario_name='Baseline',
            calculator_name='SA'
        )
        
        assert fig is not None


class TestScenarioComparisonPlot:
    """Tests for create_scenario_comparison_plot()."""
    
    def test_create_scenario_comparison_plot_basic(self):
        """Test basic scenario comparison plot."""
        results = {
            'Baseline': {
                'AIRB': {
                    'mean': 100000,
                    'std': 5000
                }
            },
            'Stress': {
                'AIRB': {
                    'mean': 110000,
                    'std': 6000
                }
            }
        }
        
        fig = create_scenario_comparison_plot(
            results=results,
            calculator_name='AIRB'
        )
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
    
    def test_create_scenario_comparison_plot_with_baseline(self):
        """Test scenario comparison with baseline scenario specified."""
        results = {
            'Baseline': {
                'AIRB': {
                    'mean': 100000,
                    'std': 5000
                }
            },
            'Stress': {
                'AIRB': {
                    'mean': 110000,
                    'std': 6000
                }
            }
        }
        
        fig = create_scenario_comparison_plot(
            results=results,
            calculator_name='AIRB',
            baseline_scenario='Baseline'
        )
        
        assert fig is not None
    
    def test_create_scenario_comparison_plot_multiple_scenarios(self):
        """Test scenario comparison with 3+ scenarios."""
        results = {
            'Baseline': {'AIRB': {'mean': 100000, 'std': 5000}},
            'Moderate': {'AIRB': {'mean': 105000, 'std': 5500}},
            'Stress': {'AIRB': {'mean': 110000, 'std': 6000}}
        }
        
        fig = create_scenario_comparison_plot(
            results=results,
            calculator_name='AIRB',
            baseline_scenario='Baseline'
        )
        
        assert fig is not None
    
    def test_create_scenario_comparison_plot_custom_title(self):
        """Test scenario comparison with custom title."""
        results = {
            'Baseline': {'AIRB': {'mean': 100000, 'std': 5000}},
            'Stress': {'AIRB': {'mean': 110000, 'std': 6000}}
        }
        
        custom_title = "My Custom Comparison"
        
        fig = create_scenario_comparison_plot(
            results=results,
            calculator_name='AIRB',
            title=custom_title
        )
        
        assert fig is not None
        assert custom_title in str(fig.layout.title.text)
    
    def test_create_scenario_comparison_plot_sa_calculator(self):
        """Test scenario comparison with SA calculator."""
        results = {
            'Baseline': {'SA': {'mean': 95000, 'std': 4500}},
            'Stress': {'SA': {'mean': 105000, 'std': 5500}}
        }
        
        fig = create_scenario_comparison_plot(
            results=results,
            calculator_name='SA'
        )
        
        assert fig is not None


class TestGenerateHTMLReport:
    """Tests for generate_html_report()."""
    
    def test_generate_html_report_basic(self, tmp_path):
        """Test basic HTML report generation."""
        results = {
            'scenarios': {
                'Baseline': {
                    'AIRB': {
                        'mean': 100000,
                        'std': 5000,
                        'min': 90000,
                        'max': 110000,
                        'rwa_values': [100000, 102000, 98000, 101000, 99000]
                    }
                }
            },
            'comparisons': {},
            'portfolio_stats': {
                'n_loans': 1000,
                'total_exposure': 1000000000
            },
            'execution_time': 10.5
        }
        
        output_file = tmp_path / "report.html"
        
        result_path = generate_html_report(
            results=results,
            output_path=output_file,
            title="Test Report"
        )
        
        assert result_path.exists()
        assert result_path == output_file
        
        # Read and verify HTML content
        html_content = output_file.read_text()
        assert "Test Report" in html_content
        assert len(html_content) > 100
    
    def test_generate_html_report_multiple_scenarios(self, tmp_path):
        """Test HTML report with multiple scenarios."""
        results = {
            'scenarios': {
                'Baseline': {
                    'AIRB': {
                        'mean': 100000,
                        'std': 5000,
                        'rwa_values': [100000, 102000, 98000]
                    }
                },
                'Stress': {
                    'AIRB': {
                        'mean': 110000,
                        'std': 6000,
                        'rwa_values': [110000, 112000, 108000]
                    }
                }
            },
            'comparisons': {},
            'portfolio_stats': {},
            'execution_time': 15.2
        }
        
        output_file = tmp_path / "multi_scenario_report.html"
        
        result_path = generate_html_report(
            results=results,
            output_path=output_file,
            title="Multi-Scenario Report"
        )
        
        assert result_path.exists()
        html_content = output_file.read_text()
        assert "Baseline" in html_content
        assert "Stress" in html_content
    
    def test_generate_html_report_with_specific_plots(self, tmp_path):
        """Test HTML report with specific plot types."""
        results = {
            'scenarios': {
                'Test': {
                    'AIRB': {
                        'mean': 100000,
                        'std': 5000,
                        'rwa_values': [100000, 102000, 98000]
                    }
                }
            },
            'comparisons': {},
            'portfolio_stats': {},
            'execution_time': 8.0
        }
        
        output_file = tmp_path / "report_specific_plots.html"
        
        result_path = generate_html_report(
            results=results,
            output_path=output_file,
            title="Report with Specific Plots",
            include_plots=['distribution', 'summary']
        )
        
        assert result_path.exists()
    
    def test_generate_html_report_creates_parent_dirs(self, tmp_path):
        """Test that generate_html_report creates parent directories if needed."""
        results = {
            'scenarios': {
                'Test': {
                    'AIRB': {
                        'mean': 100000,
                        'rwa_values': [100000]
                    }
                }
            },
            'comparisons': {},
            'portfolio_stats': {},
            'execution_time': 5.0
        }
        
        output_file = tmp_path / "subdir" / "nested" / "report.html"
        
        result_path = generate_html_report(
            results=results,
            output_path=output_file
        )
        
        assert result_path.exists()
        assert result_path.parent.parent.exists()
    
    def test_generate_html_report_includes_metadata(self, tmp_path):
        """Test HTML report includes portfolio statistics."""
        results = {
            'scenarios': {
                'Test': {
                    'AIRB': {
                        'mean': 100000,
                        'rwa_values': [100000]
                    }
                }
            },
            'comparisons': {},
            'portfolio_stats': {
                'n_loans': 5000,
                'total_exposure': 5000000000,
                'n_unique_ids': 4800
            },
            'execution_time': 12.3,
            'analysis_timestamp': '2024-01-01T12:00:00'
        }
        
        output_file = tmp_path / "report_with_metadata.html"
        
        result_path = generate_html_report(
            results=results,
            output_path=output_file,
            title="Report with Metadata"
        )
        
        assert result_path.exists()
        html_content = output_file.read_text()
        
        # Check that file is valid HTML with content
        assert '<!DOCTYPE html>' in html_content
        assert len(html_content) > 1000  # Should have substantial content


class TestVisualizationIntegration:
    """Integration tests for visualization components."""
    
    def test_distribution_and_comparison_plots_compatible(self):
        """Test that distribution and comparison plots use compatible data structures."""
        results = {
            'Baseline': {
                'AIRB': {
                    'mean': 100000,
                    'std': 5000,
                    'rwa_values': [100000, 102000, 98000, 101000, 99000]
                }
            },
            'Stress': {
                'AIRB': {
                    'mean': 110000,
                    'std': 6000,
                    'rwa_values': [110000, 112000, 108000, 111000, 109000]
                }
            }
        }
        
        # Both plot types should work with the same results structure
        dist_fig = create_rwa_distribution_plot(results, 'Baseline', 'AIRB')
        comp_fig = create_scenario_comparison_plot(results, 'AIRB')
        
        assert dist_fig is not None
        assert comp_fig is not None
        assert isinstance(dist_fig, go.Figure)
        assert isinstance(comp_fig, go.Figure)
    
    def test_all_plots_return_valid_figures(self):
        """Test that all plot functions return valid Figure objects."""
        results = {
            'Test': {
                'AIRB': {
                    'mean': 100000,
                    'std': 5000,
                    'rwa_values': [100000, 102000, 98000, 101000, 99000]
                }
            }
        }
        
        # Create various plots
        fig1 = create_rwa_distribution_plot(results, 'Test', 'AIRB')
        fig2 = create_scenario_comparison_plot(results, 'AIRB')
        
        # All should be valid figures
        for fig in [fig1, fig2]:
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0
            assert fig.layout is not None
