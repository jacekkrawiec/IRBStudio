# IRBStudio API Reference

Complete reference for IRBStudio's public API.

---

## Table of Contents

1. [High-Level API](#high-level-api)
2. [Configuration](#configuration)
3. [Data Loading](#data-loading)
4. [Portfolio Simulation](#portfolio-simulation)
5. [RWA Calculators](#rwa-calculators)
6. [Integrated Analysis](#integrated-analysis)
7. [Reporting & Visualization](#reporting--visualization)
8. [Utilities](#utilities)

---

## High-Level API

### irbstudio.run_analysis()

Run a single-scenario AIRB analysis with all configured calculators.

**Signature:**
```python
def run_analysis(
    config_path: str,
    portfolio_path: str,
    n_iterations: int = 1000,
    random_seed: Optional[int] = None,
    output_dir: str = "results",
    memory_efficient: bool = False,
    application_start_date: Optional[str] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]
```

**Parameters:**
- `config_path` (str): Path to YAML configuration file
- `portfolio_path` (str): Path to portfolio CSV or Parquet file
- `n_iterations` (int, default=1000): Number of Monte Carlo iterations
- `random_seed` (int, optional): Random seed for reproducibility
- `output_dir` (str, default="results"): Output directory for results and dashboard
- `memory_efficient` (bool, default=False): If True, discards intermediate results to save memory
- `application_start_date` (str, optional): Date to split historical vs. application data (ISO format)

**Returns:**
- `dict`: Results dictionary with structure:
  ```python
  {
      'Scenario Name': {
          'AIRB': {
              'rwa_values': np.ndarray,
              'mean': float,
              'std': float,
              'median': float,
              'percentiles': {'P5': float, 'P95': float, 'P99': float}
          },
          'SA': { ... }
      }
  }
  ```

**Example:**
```python
from irbstudio import run_analysis

results = run_analysis(
    config_path="config.yaml",
    portfolio_path="portfolio.csv",
    n_iterations=1000,
    random_seed=42,
    output_dir="results"
)

print(f"Mean AIRB RWA: ${results['Baseline']['AIRB']['mean']:,.0f}")
```

---

### irbstudio.run_scenario_comparison()

Run multiple scenarios and generate comparison dashboard.

**Signature:**
```python
def run_scenario_comparison(
    config_path: str,
    portfolio_path: str,
    n_iterations: int = 1000,
    random_seed: Optional[int] = None,
    output_dir: str = "results",
    application_start_date: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters:**
- Same as `run_analysis()`, but `memory_efficient` is always False for comparison

**Returns:**
- `dict`: Extended results dictionary including:
  ```python
  {
      'Scenario 1': { 'AIRB': {...}, 'SA': {...} },
      'Scenario 2': { 'AIRB': {...}, 'SA': {...} },
      'capital_delta': float  # Capital savings (Scenario 1 - Scenario 2) * 0.08
  }
  ```

**Example:**
```python
from irbstudio import run_scenario_comparison

results = run_scenario_comparison(
    config_path="config.yaml",
    portfolio_path="portfolio.csv",
    n_iterations=5000,
    random_seed=42
)

print(f"Capital Savings: ${results['capital_delta']:,.0f}")
```

---

### irbstudio.load_config()

Load and validate configuration from YAML file.

**Signature:**
```python
def load_config(config_path: str) -> Config
```

**Parameters:**
- `config_path` (str): Path to YAML configuration file

**Returns:**
- `Config`: Validated configuration object

**Raises:**
- `ValidationError`: If configuration is invalid
- `FileNotFoundError`: If configuration file not found

**Example:**
```python
from irbstudio import load_config

config = load_config("config.yaml")
print(f"Number of scenarios: {len(config.scenarios)}")
print(f"First scenario: {config.scenarios[0].name}")
```

---

## Configuration

### Config

Main configuration class with Pydantic validation.

**Attributes:**
- `column_mapping` (ColumnMapping): Maps portfolio columns to canonical names
- `regulatory` (RegulatoryParams): Regulatory parameters
- `scenarios` (List[Scenario]): List of scenario configurations

**Example:**
```python
from irbstudio.config.schema import Config, Scenario, ColumnMapping, RegulatoryParams

config = Config(
    column_mapping=ColumnMapping(
        loan_id='loan_id',
        exposure='balance',
        pd='pd',
        score='score',
        rating='rating',
        date='reporting_date',
        default_flag='default_flag',
        into_default_flag='into_default_flag',
        ltv='ltv'
    ),
    regulatory=RegulatoryParams(
        jurisdiction='generic',
        asset_correlation=0.15,
        confidence_level=0.999
    ),
    scenarios=[
        Scenario(
            name='Baseline',
            description='Baseline scenario',
            pd_auc=0.75,
            portfolio_default_rate=0.03,
            lgd=0.25,
            new_loan_rate=0.10,
            rating_pd_map={'A': 0.01, 'B': 0.05}
        )
    ]
)
```

---

### ColumnMapping

Maps portfolio data columns to canonical field names.

**Attributes:**
- `loan_id` (str): Unique loan identifier column
- `exposure` (str): Exposure amount column
- `pd` (str): Probability of Default column
- `score` (str): Credit score column
- `rating` (str): Rating grade column
- `date` (str): Reporting date column
- `default_flag` (str): Default status column
- `into_default_flag` (str): New default indicator column
- `ltv` (str, optional): Loan-to-Value ratio column

---

### RegulatoryParams

Regulatory parameters for RWA calculation.

**Attributes:**
- `jurisdiction` (str, default='generic'): Regulatory jurisdiction
- `asset_correlation` (float, default=0.15): Basel asset correlation parameter
- `confidence_level` (float, default=0.999): Capital confidence level

---

### Scenario

Defines a simulation scenario.

**Attributes:**
- `name` (str): Scenario name
- `description` (str, optional): Scenario description
- `pd_auc` (float): Target AUC for PD model (0.5-1.0)
- `portfolio_default_rate` (float): Portfolio-level default rate (0-1)
- `lgd` (float): Loss Given Default (0-1)
- `new_loan_rate` (float): Proportion of new originations (0-1)
- `rating_pd_map` (Dict[str, float]): Mapping of rating grades to PD values

**Validation:**
- `pd_auc` must be between 0.5 and 1.0
- All rate parameters must be between 0 and 1
- `rating_pd_map` values must be between 0 and 1

---

## Data Loading

### load_portfolio()

Load portfolio data from CSV or Parquet file.

**Signature:**
```python
def load_portfolio(
    filepath: str,
    column_mapping: Union[ColumnMapping, dict],
    parse_dates: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `filepath` (str): Path to portfolio file (CSV or Parquet)
- `column_mapping` (ColumnMapping or dict): Column name mappings
- `parse_dates` (bool, default=True): Whether to parse date columns

**Returns:**
- `pd.DataFrame`: Portfolio data with canonical column names

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format is unsupported (only CSV and Parquet supported)

**Example:**
```python
from irbstudio.data.loader import load_portfolio

portfolio_df = load_portfolio(
    filepath="portfolio.csv",
    column_mapping={'loan_id': 'id', 'exposure': 'balance', ...}
)
```

---

## Portfolio Simulation

### PortfolioSimulator

Main class for portfolio simulation with Beta mixture and migration matrix.

**Signature:**
```python
class PortfolioSimulator:
    def __init__(
        self,
        portfolio_df: pd.DataFrame,
        score_to_rating_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        rating_col: str = 'rating',
        loan_id_col: str = 'loan_id',
        date_col: str = 'date',
        default_col: str = 'default_flag',
        into_default_flag_col: str = 'into_default_flag',
        score_col: str = 'score',
        target_auc: float = 0.75,
        em_max_iter: int = 100,
        em_tol: float = 1e-6
    )
```

**Parameters:**
- `portfolio_df` (pd.DataFrame): Portfolio data with canonical column names
- `score_to_rating_bounds` (dict, optional): Rating score boundaries
  - Format: `{'A': (0.0, 0.05), 'B': (0.05, 0.15)}`
  - If None, learned from data
- `rating_col` (str): Rating column name
- `loan_id_col` (str): Loan ID column name
- `date_col` (str): Date column name
- `default_col` (str): Default flag column name
- `into_default_flag_col` (str): Into default flag column name
- `score_col` (str): Score column name
- `target_auc` (float): Target AUC for generated scores
- `em_max_iter` (int): Maximum EM algorithm iterations
- `em_tol` (float): EM algorithm convergence tolerance

**Methods:**

#### fit_beta_mixture()

Fit Beta mixture distribution to historical PD scores.

```python
def fit_beta_mixture(
    self,
    scores: np.ndarray,
    n_components: int = 2
) -> Tuple[np.ndarray, List[Tuple[float, float]]]
```

**Returns:**
- `Tuple`: (component_weights, [(alpha1, beta1), (alpha2, beta2)])

---

#### calculate_migration_matrix()

Calculate rating transition matrix from historical data.

```python
def calculate_migration_matrix(
    self,
    historical_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[int, str]]
```

**Returns:**
- `Tuple`: (transition_matrix_df, index_to_rating_map)

---

#### generate_scores()

Generate new credit scores with target AUC.

```python
def generate_scores(
    self,
    n_samples: int,
    target_auc: float,
    default_rate: float,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters:**
- `n_samples` (int): Number of samples to generate
- `target_auc` (float): Target AUC (0.5-1.0)
- `default_rate` (float): Proportion of defaults
- `random_state` (int, optional): Random seed

**Returns:**
- `Tuple`: (scores, default_labels)

---

#### simulate_portfolio()

Simulate complete portfolio with migrations and new loans.

```python
def simulate_portfolio(
    self,
    application_start_date: str,
    new_loan_rate: float = 0.1,
    portfolio_default_rate: float = 0.03,
    target_auc: float = 0.75,
    random_state: Optional[int] = None
) -> pd.DataFrame
```

**Parameters:**
- `application_start_date` (str): Date to split historical/application data
- `new_loan_rate` (float): Proportion of new loans (0-1)
- `portfolio_default_rate` (float): Target default rate
- `target_auc` (float): Target model AUC
- `random_state` (int, optional): Random seed

**Returns:**
- `pd.DataFrame`: Simulated portfolio with updated scores and ratings

**Example:**
```python
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator

simulator = PortfolioSimulator(
    portfolio_df=portfolio_df,
    score_to_rating_bounds={'A': (0.03, 0.10), 'B': (0.10, 0.20)},
    target_auc=0.80
)

simulated_portfolio = simulator.simulate_portfolio(
    application_start_date='2024-01-01',
    new_loan_rate=0.10,
    portfolio_default_rate=0.03,
    random_state=42
)
```

---

## RWA Calculators

### BaseRWACalculator

Abstract base class for all RWA calculators.

**Methods:**

#### calculate_rwa()

Calculate RWA for a portfolio.

```python
def calculate_rwa(self, portfolio_df: pd.DataFrame) -> float
```

**Parameters:**
- `portfolio_df` (pd.DataFrame): Portfolio with required fields

**Returns:**
- `float`: Total RWA

---

#### calculate_rwa_per_loan()

Calculate RWA for each loan individually.

```python
def calculate_rwa_per_loan(self, portfolio_df: pd.DataFrame) -> pd.DataFrame
```

**Returns:**
- `pd.DataFrame`: Portfolio with 'rwa' column added

---

#### summarize_rwa()

Aggregate RWA by dimension.

```python
def summarize_rwa(
    self,
    portfolio_df: pd.DataFrame,
    breakdown: str = 'rating'
) -> pd.DataFrame
```

**Parameters:**
- `portfolio_df` (pd.DataFrame): Portfolio with RWA calculated
- `breakdown` (str): Dimension to aggregate by ('rating', 'region', 'product', etc.)

**Returns:**
- `pd.DataFrame`: Summary with columns:
  - `breakdown_value`: Value of the breakdown dimension
  - `total_exposure`: Sum of exposure
  - `total_rwa`: Sum of RWA
  - `rwa_density`: RWA / Exposure ratio

---

### AIRBMortgageCalculator

AIRB calculator for mortgage portfolios.

**Signature:**
```python
class AIRBMortgageCalculator(BaseRWACalculator):
    def __init__(
        self,
        regulatory_params: Optional[Dict[str, Any]] = None
    )
```

**Parameters:**
- `regulatory_params` (dict, optional): Regulatory parameters
  - `lgd` (float): Loss Given Default
  - `asset_correlation` (float): Asset correlation parameter
  - `confidence_level` (float): Capital confidence level
  - `maturity` (float, default=2.5): Effective maturity
  - `pd_floor` (float, default=0.0003): PD floor (0.03%)

**RWA Formula:**
```
RWA = EAD × LGD × K × 12.5
where K = [LGD × N((1-R)^(-0.5) × G(PD) + (R/(1-R))^0.5 × G(0.999)) - PD × LGD] × (1-1.5×b)^(-1) × (1+(M-2.5)×b)
```

**Example:**
```python
from irbstudio.engine.mortgage import AIRBMortgageCalculator

calculator = AIRBMortgageCalculator(
    regulatory_params={
        'lgd': 0.25,
        'asset_correlation': 0.15,
        'confidence_level': 0.999
    }
)

rwa = calculator.calculate_rwa(portfolio_df)
print(f"Total AIRB RWA: ${rwa:,.0f}")
```

---

### SAMortgageCalculator

Standardized Approach calculator for mortgages.

**Signature:**
```python
class SAMortgageCalculator(BaseRWACalculator):
    def __init__(self)
```

**Risk Weights (based on LTV):**
- LTV ≤ 60%: 20%
- 60% < LTV ≤ 80%: 35%
- 80% < LTV ≤ 90%: 50%
- 90% < LTV ≤ 100%: 70%
- LTV > 100%: 100%

**RWA Formula:**
```
RWA = EAD × Risk_Weight
```

**Example:**
```python
from irbstudio.engine.mortgage import SAMortgageCalculator

calculator = SAMortgageCalculator()
rwa = calculator.calculate_rwa(portfolio_df)
print(f"Total SA RWA: ${rwa:,.0f}")
```

---

## Integrated Analysis

### IntegratedAnalysis

Orchestrates scenarios, calculators, and Monte Carlo simulation.

**Signature:**
```python
class IntegratedAnalysis:
    def __init__(self)
```

**Methods:**

#### add_calculator()

Add an RWA calculator.

```python
def add_calculator(
    self,
    name: str,
    calculator: BaseRWACalculator
) -> None
```

**Parameters:**
- `name` (str): Calculator identifier (e.g., 'AIRB', 'SA')
- `calculator` (BaseRWACalculator): Calculator instance

---

#### add_scenario()

Add a simulation scenario.

```python
def add_scenario(
    self,
    scenario_name: str,
    simulator: PortfolioSimulator,
    n_iterations: int = 1000
) -> None
```

**Parameters:**
- `scenario_name` (str): Scenario identifier
- `simulator` (PortfolioSimulator): Configured simulator
- `n_iterations` (int): Number of Monte Carlo iterations

---

#### run_scenario()

Execute a scenario with all calculators.

```python
def run_scenario(
    self,
    scenario_name: str,
    random_seed: Optional[int] = None,
    application_start_date: Optional[str] = None,
    memory_efficient: bool = False,
    process_all_dates: bool = False,
    store_full_portfolios: bool = False,
    show_progress: bool = True
) -> Dict[str, Dict[str, Any]]
```

**Parameters:**
- `scenario_name` (str): Scenario to run
- `random_seed` (int, optional): Random seed for reproducibility
- `application_start_date` (str, optional): Historical/application split date
- `memory_efficient` (bool): Discard intermediate results
- `process_all_dates` (bool): Simulate each date separately
- `store_full_portfolios` (bool): Store full simulated portfolios
- `show_progress` (bool): Show progress bar

**Returns:**
- `dict`: Results by calculator:
  ```python
  {
      'AIRB': {
          'rwa_values': np.ndarray,
          'mean': float,
          'std': float,
          'median': float,
          'percentiles': {...}
      },
      'SA': {...}
  }
  ```

---

#### get_summary_stats()

Get statistical summary for a scenario and calculator.

```python
def get_summary_stats(
    self,
    scenario_name: str,
    calculator_name: str
) -> Dict[str, float]
```

**Returns:**
- `dict`: Statistics including mean, std, median, skewness, kurtosis, VaR

---

#### get_percentiles()

Get percentile values.

```python
def get_percentiles(
    self,
    scenario_name: str,
    calculator_name: str,
    percentiles: List[float] = [5, 25, 50, 75, 95, 99]
) -> Dict[str, float]
```

**Returns:**
- `dict`: Percentile values (e.g., {'P5': ..., 'P95': ...})

---

#### compare_scenarios()

Compare two scenarios statistically.

```python
def compare_scenarios(
    self,
    scenario1: str,
    scenario2: str,
    calculator_name: str
) -> Dict[str, Any]
```

**Returns:**
- `dict`: Comparison metrics:
  - `mean_diff`: Difference in means
  - `std_diff`: Difference in standard deviations
  - `percentile_shifts`: Changes in percentiles
  - `capital_delta`: Capital impact (mean_diff × 0.08)

---

#### summarize_rwa()

Get RWA breakdown by dimension.

```python
def summarize_rwa(
    self,
    scenario_name: str,
    calculator_name: str,
    breakdown: str = 'rating'
) -> pd.DataFrame
```

**Example:**
```python
from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.engine.mortgage import AIRBMortgageCalculator

analysis = IntegratedAnalysis()
analysis.add_calculator('AIRB', AIRBMortgageCalculator())
analysis.add_scenario('Baseline', simulator, n_iterations=1000)

results = analysis.run_scenario('Baseline', random_seed=42)
stats = analysis.get_summary_stats('Baseline', 'AIRB')
percentiles = analysis.get_percentiles('Baseline', 'AIRB')

print(f"Mean RWA: ${stats['mean']:,.0f}")
print(f"P95 RWA: ${percentiles['P95']:,.0f}")
```

---

## Reporting & Visualization

### create_dashboard()

Generate interactive HTML dashboard with Plotly.

**Signature:**
```python
def create_dashboard(
    analysis_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_path: str = "dashboard.html",
    title: str = "AIRB Scenario Analysis Dashboard"
) -> str
```

**Parameters:**
- `analysis_results` (dict): Results from IntegratedAnalysis
- `output_path` (str): Output file path
- `title` (str): Dashboard title

**Returns:**
- `str`: Path to generated HTML file

**Dashboard Sections:**
1. Distribution plots for each scenario
2. Scenario comparison charts
3. Statistical summary tables
4. Percentile waterfall charts

**Example:**
```python
from irbstudio.reporting.dashboard import create_dashboard

html_path = create_dashboard(
    analysis_results=analysis.results,
    output_path="results/dashboard.html",
    title="Q4 2024 Capital Impact Analysis"
)
```

---

### create_distribution_plot()

Create RWA distribution plot for a single scenario.

**Signature:**
```python
def create_distribution_plot(
    rwa_values: np.ndarray,
    scenario_name: str = "Scenario",
    calculator_name: str = "AIRB"
) -> go.Figure
```

**Returns:**
- `plotly.graph_objects.Figure`: Interactive histogram with KDE overlay

---

### create_scenario_comparison_plot()

Create side-by-side comparison of scenarios.

**Signature:**
```python
def create_scenario_comparison_plot(
    results: Dict[str, Dict[str, Any]],
    calculator_name: str = "AIRB"
) -> go.Figure
```

**Requirements:**
- `results` must include 'mean', 'std', 'median' keys for each scenario

**Returns:**
- `plotly.graph_objects.Figure`: Grouped bar chart comparing scenarios

---

### create_waterfall_chart()

Create waterfall chart showing RWA components or changes.

**Signature:**
```python
def create_waterfall_chart(
    data: Dict[str, float],
    title: str = "RWA Waterfall"
) -> go.Figure
```

**Parameters:**
- `data` (dict): Component values (e.g., {'Base': 1000, 'Delta 1': -100, 'Net': 900})
- `title` (str): Chart title

**Returns:**
- `plotly.graph_objects.Figure`: Waterfall chart

---

## Utilities

### setup_logging()

Configure centralized logging.

**Signature:**
```python
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger
```

**Parameters:**
- `level` (str): Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
- `log_file` (str, optional): Path to log file
- `format` (str): Log message format

**Returns:**
- `logging.Logger`: Configured logger

**Example:**
```python
from irbstudio.utils.logging import setup_logging

logger = setup_logging(level="DEBUG", log_file="analysis.log")
logger.info("Starting analysis...")
```

---

## Type Hints

IRBStudio uses type hints throughout. Key types:

```python
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

# Common type aliases
RWAResults = Dict[str, Dict[str, Any]]
ScenarioResults = Dict[str, Dict[str, Dict[str, Any]]]
RatingPDMap = Dict[str, float]
RatingBounds = Dict[str, Tuple[float, float]]
```

---

## Error Handling

### Common Exceptions

**ValidationError** (from Pydantic)
- Raised when configuration validation fails
- Contains detailed error messages

**ValueError**
- Invalid parameter values
- Unsupported file formats
- Empty or invalid data

**KeyError**
- Missing required columns
- Undefined scenario or calculator names

**FileNotFoundError**
- Configuration or portfolio file not found

**Example Error Handling:**
```python
from pydantic import ValidationError

try:
    config = load_config("config.yaml")
except ValidationError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError:
    print("Configuration file not found")
```

---

## Best Practices

### 1. Always Set Random Seed
```python
results = analysis.run_scenario(
    scenario_name='Baseline',
    random_seed=42  # Reproducible results
)
```

### 2. Use Type Hints
```python
from typing import Dict, List
import pandas as pd

def process_portfolio(df: pd.DataFrame) -> Dict[str, float]:
    ...
```

### 3. Validate Configuration Early
```python
config = load_config("config.yaml")
# Any validation errors raised immediately
```

### 4. Enable Logging for Debugging
```python
from irbstudio.utils.logging import setup_logging

logger = setup_logging(level="DEBUG")
```

### 5. Use Context Managers for Resources
```python
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Version Compatibility

- **Python**: 3.9+
- **pandas**: 1.5+
- **numpy**: 1.20+
- **scipy**: 1.7+
- **plotly**: 5.0+
- **pydantic**: 2.0+

---

*For usage examples, see the [User Guide](user_guide.md).*

*Last Updated: January 2025*
