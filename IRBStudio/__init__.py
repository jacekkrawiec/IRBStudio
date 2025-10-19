"""
IRBStudio - AIRB Scenario & Impact Analysis Engine

A Python package for simulating and analyzing the impact of modeling choices
and parameter assumptions on Risk-Weighted Assets (RWA) and capital requirements.

Quick Start:
    >>> from irbstudio import run_analysis
    >>> 
    >>> results = run_analysis(
    ...     config_path='config.yaml',
    ...     portfolio_path='portfolio.csv',
    ...     n_iterations=1000
    ... )
"""

__version__ = "0.1.0"

# Import main API functions
from .main import (
    run_analysis,
    run_scenario_comparison,
    load_config
)

# Import key classes for advanced usage
from .simulation.portfolio_simulator import PortfolioSimulator
from .engine.integrated_analysis import IntegratedAnalysis
from .engine.mortgage.airb_calculator import AIRBMortgageCalculator
from .engine.mortgage.sa_calculator import SAMortgageCalculator

# Import configuration schemas
from .config.schema import Config, Scenario, ColumnMapping, RegulatoryParams

# Import data loading utilities
from .data.loader import load_portfolio

__all__ = [
    # Main API
    'run_analysis',
    'run_scenario_comparison',
    'load_config',
    
    # Core classes
    'PortfolioSimulator',
    'IntegratedAnalysis',
    'AIRBMortgageCalculator',
    'SAMortgageCalculator',
    
    # Configuration
    'Config',
    'Scenario',
    'ColumnMapping',
    'RegulatoryParams',
    
    # Data loading
    'load_portfolio',
    
    # Metadata
    '__version__',
]
