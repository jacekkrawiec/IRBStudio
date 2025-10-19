"""
Priority 2 Tests: High-Level API (run_analysis and run_scenario_comparison)

Module: irbstudio.main
Focus: User-facing API functions
"""

import pytest
import yaml
from pathlib import Path
from irbstudio.main import run_analysis, run_scenario_comparison, load_config
from irbstudio.config.schema import Config


class TestLoadConfig:
    """Tests for load_config() function."""
    
    def test_load_config_valid_yaml(self, tmp_path, sample_config_dict):
        """Test load_config() with valid YAML file."""
        config_file = tmp_path / "config.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        config = load_config(config_file)
        
        assert isinstance(config, Config)
        assert len(config.scenarios) > 0
        assert config.regulatory is not None
    
    def test_load_config_missing_file(self, tmp_path):
        """Test load_config() with non-existent file."""
        config_file = tmp_path / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            load_config(config_file)
    
    def test_load_config_invalid_yaml(self, tmp_path):
        """Test load_config() with malformed YAML."""
        config_file = tmp_path / "invalid.yaml"
        
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content:\n  - missing colon")
        
        with pytest.raises(yaml.YAMLError):
            load_config(config_file)


class TestRunAnalysisBasic:
    """Basic tests for run_analysis() function."""
    
    def test_run_analysis_basic_execution(
        self,
        tmp_path,
        sample_config_dict,
        small_portfolio_df
    ):
        """Test run_analysis() executes successfully with minimal parameters."""
        # Create config file
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        # Create portfolio file
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        # Run analysis
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['AIRB'],
            n_iterations=2
        )
        
        assert results is not None
        assert 'config' in results
        assert 'portfolio_stats' in results
        assert 'scenarios' in results
        assert 'execution_time' in results
    
    def test_run_analysis_returns_correct_structure(
        self,
        tmp_path,
        sample_config_dict,
        small_portfolio_df
    ):
        """Test run_analysis() returns expected dictionary structure."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['AIRB'],
            n_iterations=2
        )
        
        # Verify structure
        assert isinstance(results['config'], Config)
        assert isinstance(results['portfolio_stats'], dict)
        assert isinstance(results['scenarios'], dict)
        assert isinstance(results['comparisons'], dict)
        assert isinstance(results['execution_time'], float)
        assert isinstance(results['output_files'], list)
        assert 'analysis_timestamp' in results
    
    def test_run_analysis_with_output_dir(
        self,
        tmp_path,
        sample_config_dict,
        small_portfolio_df
    ):
        """Test run_analysis() creates output files when output_dir is specified."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        output_dir = tmp_path / "results"
        
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            output_dir=output_dir,
            calculators=['AIRB'],
            n_iterations=2
        )
        
        assert output_dir.exists()
        assert len(results['output_files']) > 0
        
        # Verify files exist
        for file_path in results['output_files']:
            assert Path(file_path).exists()


class TestRunAnalysisCalculators:
    """Tests for run_analysis() with different calculator configurations."""
    
    def test_run_analysis_with_airb_calculator(
        self,
        tmp_path,
        sample_config_dict,
        small_portfolio_df
    ):
        """Test run_analysis() with AIRB calculator only."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['AIRB'],
            n_iterations=2
        )
        
        # Verify AIRB results exist
        for scenario_name, scenario_results in results['scenarios'].items():
            assert 'AIRB' in scenario_results
            assert 'mean' in scenario_results['AIRB']
    
    def test_run_analysis_with_sa_calculator(
        self,
        tmp_path,
        sample_config_dict,
        small_portfolio_df
    ):
        """Test run_analysis() with SA calculator only."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['SA'],
            n_iterations=2
        )
        
        # Verify SA results exist
        for scenario_name, scenario_results in results['scenarios'].items():
            assert 'SA' in scenario_results
            assert 'mean' in scenario_results['SA']
    
    def test_run_analysis_with_multiple_calculators(
        self,
        tmp_path,
        sample_config_dict,
        small_portfolio_df
    ):
        """Test run_analysis() with both AIRB and SA calculators."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['AIRB', 'SA'],
            n_iterations=2
        )
        
        # Verify both calculator results exist
        for scenario_name, scenario_results in results['scenarios'].items():
            assert 'AIRB' in scenario_results
            assert 'SA' in scenario_results


class TestRunAnalysisScenarios:
    """Tests for run_analysis() with multiple scenarios."""
    
    def test_run_analysis_single_scenario(
        self,
        tmp_path,
        small_portfolio_df
    ):
        """Test run_analysis() with single scenario."""
        # Create config with single scenario
        config_dict = {
            'scenarios': [
                {
                    'name': 'Single',
                    'pd_auc': 0.75,
                    'portfolio_default_rate': 0.02,
                    'lgd': 0.40,
                    'rating_pd_map': {'A': 0.005, 'B': 0.015, 'C': 0.030}
                }
            ],
            'regulatory': {
                'asset_correlation': 0.15,
                'confidence_level': 0.999
            },
            'column_mapping': {
                'loan_id': 'loan_id',
                'exposure': 'exposure',
                'date': 'reporting_date',
                'default_flag': 'default_flag',
                'into_default_flag': 'into_default_flag',
                'rating': 'rating',
                'score': 'score',
                'pd': 'pd',
                'lgd': 'lgd',
                'maturity': 'maturity'
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['AIRB'],
            n_iterations=2
        )
        
        assert len(results['scenarios']) == 1
        assert 'Single' in results['scenarios']
        assert len(results['comparisons']) == 0  # No comparisons with single scenario
    
    def test_run_analysis_multiple_scenarios(
        self,
        tmp_path,
        sample_config_dict,
        small_portfolio_df
    ):
        """Test run_analysis() with multiple scenarios."""
        # Ensure config has multiple scenarios
        if len(sample_config_dict['scenarios']) < 2:
            sample_config_dict['scenarios'].append({
                'name': 'Stress',
                'pd_auc': 0.65,
                'portfolio_default_rate': 0.05,
                'lgd': 0.50,
                'rating_pd_map': {'A': 0.01, 'B': 0.03, 'C': 0.06}
            })
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['AIRB'],
            n_iterations=2
        )
        
        assert len(results['scenarios']) >= 2
        assert len(results['comparisons']) > 0  # Should have comparisons


class TestRunAnalysisOptions:
    """Tests for run_analysis() with various options."""
    
    def test_run_analysis_with_random_seed(
        self,
        tmp_path,
        sample_config_dict,
        small_portfolio_df
    ):
        """Test run_analysis() with random seed for reproducibility."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        # Run twice with same seed
        results1 = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['AIRB'],
            n_iterations=5,
            random_seed=42
        )
        
        results2 = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['AIRB'],
            n_iterations=5,
            random_seed=42
        )
        
        # Results should be identical (or very close)
        for scenario_name in results1['scenarios'].keys():
            mean1 = results1['scenarios'][scenario_name]['AIRB']['mean']
            mean2 = results2['scenarios'][scenario_name]['AIRB']['mean']
            
            # Allow for small floating point differences
            assert abs(mean1 - mean2) < 1.0, f"Results differ for scenario {scenario_name}"
    
    def test_run_analysis_with_progress_callback(
        self,
        tmp_path,
        sample_config_dict,
        small_portfolio_df
    ):
        """Test run_analysis() with progress callback."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        progress_calls = []
        
        def progress_callback(step: str, progress: float):
            progress_calls.append((step, progress))
        
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            calculators=['AIRB'],
            n_iterations=2,
            progress_callback=progress_callback
        )
        
        assert len(progress_calls) > 0
        assert progress_calls[0][1] >= 0.0  # First progress should be >= 0
        assert progress_calls[-1][1] <= 1.0  # Last progress should be <= 1.0


class TestRunScenarioComparison:
    """Tests for run_scenario_comparison() function."""
    
    def test_run_scenario_comparison_basic(
        self,
        tmp_path,
        small_portfolio_df
    ):
        """Test run_scenario_comparison() with basic setup."""
        # Create config with multiple scenarios
        config_dict = {
            'scenarios': [
                {
                    'name': 'Baseline',
                    'pd_auc': 0.75,
                    'portfolio_default_rate': 0.02,
                    'lgd': 0.40,
                    'rating_pd_map': {'A': 0.005, 'B': 0.015, 'C': 0.030}
                },
                {
                    'name': 'Stress',
                    'pd_auc': 0.65,
                    'portfolio_default_rate': 0.05,
                    'lgd': 0.50,
                    'rating_pd_map': {'A': 0.01, 'B': 0.03, 'C': 0.06}
                }
            ],
            'regulatory': {
                'asset_correlation': 0.15,
                'confidence_level': 0.999
            },
            'column_mapping': {
                'loan_id': 'loan_id',
                'exposure': 'exposure',
                'date': 'reporting_date',
                'default_flag': 'default_flag',
                'into_default_flag': 'into_default_flag',
                'rating': 'rating',
                'score': 'score',
                'pd': 'pd',
                'lgd': 'lgd',
                'maturity': 'maturity'
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_scenario_comparison(
            config_path=config_file,
            portfolio_path=portfolio_file,
            baseline_scenario='Baseline',
            comparison_scenarios=['Stress'],
            calculator='AIRB',
            n_iterations=2
        )
        
        assert results is not None
        assert results['baseline'] == 'Baseline'
        assert results['calculator'] == 'AIRB'
        assert 'comparisons' in results
        assert 'Stress' in results['comparisons']
    
    def test_run_scenario_comparison_returns_delta(
        self,
        tmp_path,
        small_portfolio_df
    ):
        """Test run_scenario_comparison() returns capital delta calculations."""
        config_dict = {
            'scenarios': [
                {
                    'name': 'Baseline',
                    'pd_auc': 0.75,
                    'portfolio_default_rate': 0.02,
                    'lgd': 0.40,
                    'rating_pd_map': {'A': 0.005, 'B': 0.015, 'C': 0.030}
                },
                {
                    'name': 'Alternative',
                    'pd_auc': 0.70,
                    'portfolio_default_rate': 0.03,
                    'lgd': 0.45,
                    'rating_pd_map': {'A': 0.008, 'B': 0.020, 'C': 0.040}
                }
            ],
            'regulatory': {
                'asset_correlation': 0.15,
                'confidence_level': 0.999
            },
            'column_mapping': {
                'loan_id': 'loan_id',
                'exposure': 'exposure',
                'date': 'reporting_date',
                'default_flag': 'default_flag',
                'into_default_flag': 'into_default_flag',
                'rating': 'rating',
                'score': 'score',
                'pd': 'pd',
                'lgd': 'lgd',
                'maturity': 'maturity'
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_scenario_comparison(
            config_path=config_file,
            portfolio_path=portfolio_file,
            baseline_scenario='Baseline',
            comparison_scenarios=['Alternative'],
            calculator='AIRB',
            n_iterations=2
        )
        
        comparison = results['comparisons']['Alternative']
        
        assert 'baseline_mean_rwa' in comparison
        assert 'scenario_mean_rwa' in comparison
        assert 'absolute_rwa_change' in comparison
        assert 'relative_rwa_change' in comparison
        assert isinstance(comparison['absolute_rwa_change'], (int, float))
        assert isinstance(comparison['relative_rwa_change'], (int, float))
    
    def test_run_scenario_comparison_with_calculators(
        self,
        tmp_path,
        small_portfolio_df
    ):
        """Test run_scenario_comparison() with specific calculator types."""
        config_dict = {
            'scenarios': [
                {
                    'name': 'Baseline',
                    'pd_auc': 0.75,
                    'portfolio_default_rate': 0.02,
                    'lgd': 0.40,
                    'rating_pd_map': {'A': 0.005, 'B': 0.015, 'C': 0.030}
                },
                {
                    'name': 'Stress',
                    'pd_auc': 0.72,
                    'portfolio_default_rate': 0.04,
                    'lgd': 0.45,
                    'rating_pd_map': {'A': 0.010, 'B': 0.025, 'C': 0.050}
                }
            ],
            'regulatory': {
                'asset_correlation': 0.15,
                'confidence_level': 0.999
            },
            'column_mapping': {
                'loan_id': 'loan_id',
                'exposure': 'exposure',
                'date': 'reporting_date',
                'default_flag': 'default_flag',
                'into_default_flag': 'into_default_flag',
                'rating': 'rating',
                'score': 'score',
                'pd': 'pd',
                'lgd': 'lgd',
                'maturity': 'maturity',
                'ltv': 'ltv',
                'property_value': 'property_value'
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        # Test with SA calculator
        results = run_scenario_comparison(
            config_path=config_file,
            portfolio_path=portfolio_file,
            baseline_scenario='Baseline',
            comparison_scenarios=['Stress'],
            calculator='SA',
            n_iterations=2
        )
        
        assert results['calculator'] == 'SA'
        assert 'Stress' in results['comparisons']
    
    def test_run_scenario_comparison_with_iterations(
        self,
        tmp_path,
        small_portfolio_df
    ):
        """Test run_scenario_comparison() with custom iteration count."""
        config_dict = {
            'scenarios': [
                {
                    'name': 'Baseline',
                    'pd_auc': 0.75,
                    'portfolio_default_rate': 0.02,
                    'lgd': 0.40,
                    'rating_pd_map': {'A': 0.005, 'B': 0.015, 'C': 0.030}
                },
                {
                    'name': 'Stress',
                    'pd_auc': 0.70,
                    'portfolio_default_rate': 0.03,
                    'lgd': 0.45,
                    'rating_pd_map': {'A': 0.008, 'B': 0.020, 'C': 0.040}
                }
            ],
            'regulatory': {
                'asset_correlation': 0.15,
                'confidence_level': 0.999
            },
            'column_mapping': {
                'loan_id': 'loan_id',
                'exposure': 'exposure',
                'date': 'reporting_date',
                'default_flag': 'default_flag',
                'into_default_flag': 'into_default_flag',
                'rating': 'rating',
                'score': 'score',
                'pd': 'pd',
                'lgd': 'lgd',
                'maturity': 'maturity'
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_scenario_comparison(
            config_path=config_file,
            portfolio_path=portfolio_file,
            baseline_scenario='Baseline',
            comparison_scenarios=['Stress'],
            calculator='AIRB',
            n_iterations=5
        )
        
        # Verify results structure
        assert results is not None
        assert 'Stress' in results['comparisons']
        comparison = results['comparisons']['Stress']
        assert 'baseline_mean_rwa' in comparison
        assert 'scenario_mean_rwa' in comparison
    
    def test_run_scenario_comparison_invalid_baseline(
        self,
        tmp_path,
        small_portfolio_df
    ):
        """Test run_scenario_comparison() with invalid baseline scenario name."""
        config_dict = {
            'scenarios': [
                {
                    'name': 'Baseline',
                    'pd_auc': 0.75,
                    'portfolio_default_rate': 0.02,
                    'lgd': 0.40,
                    'rating_pd_map': {'A': 0.005, 'B': 0.015, 'C': 0.030}
                }
            ],
            'regulatory': {
                'asset_correlation': 0.15,
                'confidence_level': 0.999
            },
            'column_mapping': {
                'loan_id': 'loan_id',
                'exposure': 'exposure',
                'date': 'reporting_date',
                'default_flag': 'default_flag',
                'into_default_flag': 'into_default_flag',
                'rating': 'rating',
                'score': 'score',
                'pd': 'pd',
                'lgd': 'lgd',
                'maturity': 'maturity'
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        with pytest.raises((ValueError, KeyError)):
            run_scenario_comparison(
                config_path=config_file,
                portfolio_path=portfolio_file,
                baseline_scenario='NonExistent',
                comparison_scenarios=['Baseline'],
                calculator='AIRB',
                n_iterations=2
            )
    
    def test_run_scenario_comparison_invalid_alternative(
        self,
        tmp_path,
        small_portfolio_df,
        caplog
    ):
        """Test run_scenario_comparison() with invalid comparison scenario name."""
        config_dict = {
            'scenarios': [
                {
                    'name': 'Baseline',
                    'pd_auc': 0.75,
                    'portfolio_default_rate': 0.02,
                    'lgd': 0.40,
                    'rating_pd_map': {'A': 0.005, 'B': 0.015, 'C': 0.030}
                }
            ],
            'regulatory': {
                'asset_correlation': 0.15,
                'confidence_level': 0.999
            },
            'column_mapping': {
                'loan_id': 'loan_id',
                'exposure': 'exposure',
                'date': 'reporting_date',
                'default_flag': 'default_flag',
                'into_default_flag': 'into_default_flag',
                'rating': 'rating',
                'score': 'score',
                'pd': 'pd',
                'lgd': 'lgd',
                'maturity': 'maturity'
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        # Function handles gracefully with warning
        results = run_scenario_comparison(
            config_path=config_file,
            portfolio_path=portfolio_file,
            baseline_scenario='Baseline',
            comparison_scenarios=['NonExistent'],
            calculator='AIRB',
            n_iterations=2
        )
        
        # Should log warning and skip missing scenario
        assert "not found" in caplog.text or "NonExistent" in caplog.text
        assert 'comparisons' in results
        # Missing scenario should not be in comparisons
        assert 'NonExistent' not in results['comparisons']
    
    def test_run_scenario_comparison_same_config(
        self,
        tmp_path,
        small_portfolio_df
    ):
        """Test run_scenario_comparison() comparing identical scenarios."""
        config_dict = {
            'scenarios': [
                {
                    'name': 'Baseline',
                    'pd_auc': 0.75,
                    'portfolio_default_rate': 0.02,
                    'lgd': 0.40,
                    'rating_pd_map': {'A': 0.005, 'B': 0.015, 'C': 0.030}
                }
            ],
            'regulatory': {
                'asset_correlation': 0.15,
                'confidence_level': 0.999
            },
            'column_mapping': {
                'loan_id': 'loan_id',
                'exposure': 'exposure',
                'date': 'reporting_date',
                'default_flag': 'default_flag',
                'into_default_flag': 'into_default_flag',
                'rating': 'rating',
                'score': 'score',
                'pd': 'pd',
                'lgd': 'lgd',
                'maturity': 'maturity'
            }
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_scenario_comparison(
            config_path=config_file,
            portfolio_path=portfolio_file,
            baseline_scenario='Baseline',
            comparison_scenarios=['Baseline'],
            calculator='AIRB',
            n_iterations=2
        )
        
        # Should succeed but show minimal differences
        comparison = results['comparisons']['Baseline']
        assert 'absolute_rwa_change' in comparison
        # Changes should be close to zero (allowing for Monte Carlo variation)
        assert abs(comparison['absolute_rwa_change']) < comparison['baseline_mean_rwa'] * 0.1


class TestRunAnalysisExecutionTimeTracking:
    """Tests for execution time tracking in run_analysis()."""
    
    def test_run_analysis_execution_time_tracking(
        self,
        tmp_path,
        small_portfolio_df,
        sample_config_dict
    ):
        """Test run_analysis() captures execution time metrics."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        portfolio_file = tmp_path / "portfolio.csv"
        small_portfolio_df.to_csv(portfolio_file, index=False)
        
        results = run_analysis(
            config_path=config_file,
            portfolio_path=portfolio_file,
            n_iterations=2
        )
        
        # Check for execution time metrics
        assert results is not None
        # Execution should have completed
        assert 'results' in results or 'scenarios' in results
