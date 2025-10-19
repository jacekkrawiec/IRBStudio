# IRBStudio User Guide

Welcome to the IRBStudio User Guide! This comprehensive guide will help you understand and use IRBStudio for AIRB scenario analysis and capital impact calculations.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Quick Start Tutorial](#quick-start-tutorial)
5. [Configuration Guide](#configuration-guide)
6. [Data Preparation](#data-preparation)
7. [Running Analyses](#running-analyses)
8. [Understanding Results](#understanding-results)
9. [Advanced Usage](#advanced-usage)
10. [Real-World Examples](#real-world-examples)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

---

## Introduction

### What is IRBStudio?

IRBStudio is a Python-based scenario analysis engine designed for internal risk analysts and model owners in banks using the Internal Ratings-Based (IRB) Approach under Basel III/IV regulations.

**Key Capabilities:**
- Simulate "what-if" scenarios to understand capital impacts
- Compare AIRB vs. Standardized Approach RWA
- Analyze the effect of model improvements (e.g., higher AUC)
- Generate interactive visualizations and reports
- Run Monte Carlo simulations for distributional insights

### Who Should Use IRBStudio?

- **Risk Analysts**: Performing capital impact assessments
- **Model Validators**: Understanding model risk and uncertainty
- **Model Owners**: Evaluating model improvements and development priorities
- **Risk Managers**: Strategic capital planning and optimization

### What Problems Does It Solve?

1. **Capital Planning**: Quantify the RWA and capital impact of strategic decisions
2. **Model Development**: Prioritize model improvements based on capital impact
3. **Scenario Analysis**: Compare multiple modeling approaches quickly
4. **Regulatory Analysis**: Understand AIRB vs. SA trade-offs
5. **Documentation**: Generate reproducible, auditable analysis reports

---

## Installation

### Prerequisites

- **Python 3.9 or higher**
- **pip** package manager
- A virtual environment tool (recommended)

### Step-by-Step Installation

#### 1. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv irbenv
.\irbenv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv irbenv
source irbenv/bin/activate
```

#### 2. Install IRBStudio

**From Source (Development):**
```bash
git clone https://github.com/jacekkrawiec/IRBStudio.git
cd IRBStudio
pip install -e .
```

**From PyPI (Coming Soon):**
```bash
pip install irbstudio
```

#### 3. Verify Installation

```python
import irbstudio
print(irbstudio.__version__)
```

---

## Core Concepts

### 1. Portfolio Simulation

IRBStudio uses a hybrid approach to simulate realistic portfolio behavior:

**Components:**
- **Beta Mixture Model**: Learns the distribution of PD scores from historical data
- **Migration Matrix**: Captures rating transitions over time
- **AUC-Driven Generation**: Generates new scores with target discriminatory power
- **Monte Carlo**: Produces full RWA distributions (not just point estimates)

**Key Innovation**: The simulator correctly segments portfolios into:
- **Historical vs. Application**: Existing loans vs. new originations
- **Existing vs. New Clients**: Repeat customers vs. first-time borrowers

### 2. RWA Calculators

IRBStudio implements multiple approaches:

**AIRB (Advanced Internal Ratings-Based):**
- Uses bank's internal PD estimates
- Regulatory LGD and maturity adjustments
- Asset correlation based on PD
- Full regulatory formula implementation

**SA (Standardized Approach):**
- Regulatory risk weights based on LTV
- Simpler calculation, higher capital requirements
- Used as baseline for comparison

### 3. Scenarios

A **scenario** represents a specific set of assumptions:

```python
Scenario(
    name="Improved Model",
    description="PD model with higher AUC",
    pd_auc=0.85,              # Target AUC (discriminatory power)
    portfolio_default_rate=0.03,  # Overall default rate
    lgd=0.25,                 # Loss Given Default
    new_loan_rate=0.10,       # Proportion of new originations
    rating_pd_map={           # Rating grade PD mapping
        'AAA': 0.001,
        'AA': 0.005,
        'A': 0.01,
        'BBB': 0.03,
        'BB': 0.05,
        'B': 0.10
    }
)
```

### 4. Integrated Analysis

The `IntegratedAnalysis` class orchestrates the complete workflow:

```
Portfolio Data → Simulator → RWA Calculators → Results → Visualizations
```

It handles:
- Multiple scenarios
- Multiple calculators
- Result aggregation
- Statistical analysis
- Report generation

---

## Quick Start Tutorial

### Example 1: Basic AIRB Analysis

This example demonstrates a complete AIRB analysis workflow.

#### Step 1: Create Portfolio Data

Create `my_portfolio.csv`:
```csv
loan_id,balance,pd,score,rating,reporting_date,default_flag,into_default_flag,ltv,property_value
L001,250000,0.02,0.05,A,2024-01-01,0,0,0.75,333333
L002,180000,0.05,0.12,B,2024-01-01,0,0,0.80,225000
L003,320000,0.03,0.08,A,2024-01-01,0,0,0.70,457143
L004,150000,0.08,0.15,BB,2024-01-01,0,0,0.85,176471
L005,200000,0.04,0.09,BBB,2024-01-01,0,0,0.78,256410
L006,280000,0.02,0.06,A,2024-06-01,0,0,0.72,388889
L007,190000,0.06,0.13,B,2024-06-01,0,0,0.82,231707
L008,350000,0.01,0.03,AA,2024-06-01,0,0,0.65,538462
```

#### Step 2: Create Configuration

Create `config.yaml`:
```yaml
# Map your data columns to canonical fields
column_mapping:
  loan_id: loan_id
  exposure: balance
  pd: pd
  score: score
  rating: rating
  date: reporting_date
  default_flag: default_flag
  into_default_flag: into_default_flag
  ltv: ltv

# Regulatory parameters
regulatory:
  jurisdiction: generic
  asset_correlation: 0.15
  confidence_level: 0.999

# Define scenarios
scenarios:
  - name: "Current State"
    description: "Current model performance (AUC=0.75)"
    pd_auc: 0.75
    portfolio_default_rate: 0.03
    lgd: 0.25
    new_loan_rate: 0.10
    rating_pd_map:
      AA: 0.005
      A: 0.01
      BBB: 0.03
      BB: 0.05
      B: 0.10

  - name: "Target State"
    description: "Improved model (AUC=0.85)"
    pd_auc: 0.85
    portfolio_default_rate: 0.03
    lgd: 0.25
    new_loan_rate: 0.10
    rating_pd_map:
      AA: 0.005
      A: 0.01
      BBB: 0.03
      BB: 0.05
      B: 0.10
```

#### Step 3: Run Analysis

```python
from irbstudio import run_scenario_comparison

# Run comparison
results = run_scenario_comparison(
    config_path="config.yaml",
    portfolio_path="my_portfolio.csv",
    n_iterations=1000,
    random_seed=42,
    output_dir="results"
)

# Print summary
print("\n=== RWA Comparison ===")
for scenario_name, scenario_results in results.items():
    if scenario_name != 'capital_delta':
        airb_mean = scenario_results['AIRB']['mean']
        print(f"{scenario_name}: ${airb_mean:,.0f}")

print(f"\nCapital Savings: ${results['capital_delta']:,.0f}")
print("\nDashboard saved to: results/scenario_comparison_dashboard.html")
```

#### Step 4: View Results

Open `results/scenario_comparison_dashboard.html` in your browser to see:
- RWA distribution plots for each scenario
- Side-by-side comparison charts
- Statistical summaries
- Percentile analysis

---

## Configuration Guide

### Complete Configuration Template

```yaml
# ============================================
# COLUMN MAPPING
# ============================================
# Map your portfolio data columns to IRBStudio's canonical field names
column_mapping:
  loan_id: loan_id                # Unique identifier for each loan
  exposure: balance               # Exposure amount (EAD)
  pd: pd                          # Probability of Default
  score: score                    # Credit score (0-1 scale)
  rating: rating                  # Internal rating grade
  date: reporting_date            # Reporting/observation date
  default_flag: default_flag      # 1 if loan is in default, 0 otherwise
  into_default_flag: into_default_flag  # 1 if loan defaulted in this period
  ltv: ltv                        # Loan-to-Value ratio (for SA calculator)

# ============================================
# REGULATORY PARAMETERS
# ============================================
regulatory:
  jurisdiction: generic           # Regulatory jurisdiction
  asset_correlation: 0.15         # Asset correlation parameter (Basel)
  confidence_level: 0.999         # Confidence level for capital calculation

# ============================================
# SCENARIOS
# ============================================
scenarios:
  - name: "Baseline"
    description: "Current state baseline scenario"
    
    # Model Performance
    pd_auc: 0.75                  # Area Under Curve for PD model
    portfolio_default_rate: 0.03  # Portfolio-level default rate
    
    # Risk Parameters
    lgd: 0.25                     # Loss Given Default (25%)
    
    # Portfolio Dynamics
    new_loan_rate: 0.10           # 10% of portfolio is new originations
    
    # Rating PD Mapping
    rating_pd_map:
      AAA: 0.0005
      AA: 0.001
      A: 0.005
      BBB: 0.01
      BB: 0.03
      B: 0.05
      CCC: 0.10
      D: 1.0

  - name: "Stress Scenario"
    description: "Adverse economic conditions"
    pd_auc: 0.70                  # Lower discrimination
    portfolio_default_rate: 0.06  # Higher defaults
    lgd: 0.35                     # Higher LGD
    new_loan_rate: 0.05           # Lower new originations
    rating_pd_map:
      AAA: 0.001
      AA: 0.002
      A: 0.01
      BBB: 0.02
      BB: 0.06
      B: 0.10
      CCC: 0.20
      D: 1.0
```

### Configuration Field Descriptions

#### Column Mapping

| Field | Required | Description |
|-------|----------|-------------|
| `loan_id` | Yes | Unique identifier for each loan |
| `exposure` | Yes | Exposure at Default (EAD) amount |
| `pd` | Yes | Probability of Default (0-1) |
| `score` | Yes | Credit score (0-1, lower = better) |
| `rating` | Yes | Internal rating grade (e.g., 'A', 'BBB') |
| `date` | Yes | Reporting/observation date |
| `default_flag` | Yes | Whether loan is currently in default |
| `into_default_flag` | Yes | Whether loan defaulted this period |
| `ltv` | For SA | Loan-to-Value ratio (for SA calculator) |

#### Regulatory Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `jurisdiction` | `generic` | Regulatory jurisdiction |
| `asset_correlation` | `0.15` | Basel asset correlation parameter |
| `confidence_level` | `0.999` | Capital confidence level (99.9%) |

#### Scenario Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | Yes | Scenario name (used in reports) |
| `description` | No | Scenario description |
| `pd_auc` | Yes | Target AUC for PD model (0.5-1.0) |
| `portfolio_default_rate` | Yes | Target portfolio default rate |
| `lgd` | Yes | Loss Given Default (0-1) |
| `new_loan_rate` | Yes | Proportion of new originations |
| `rating_pd_map` | Yes | Mapping of rating grades to PD values |

---

## Data Preparation

### Portfolio Data Requirements

#### Minimum Required Columns

Your portfolio data must include at least:

1. **loan_id**: Unique identifier
2. **balance**: Exposure amount
3. **pd**: Probability of default
4. **score**: Credit score (0-1 scale)
5. **rating**: Internal rating grade
6. **reporting_date**: Observation date
7. **default_flag**: Default status (0/1)
8. **into_default_flag**: New default indicator (0/1)

#### Additional Columns for SA Calculator

- **ltv**: Loan-to-Value ratio
- **property_value**: Property value (optional, can be calculated from balance/ltv)

### Data Format Guidelines

#### Date Format
- **ISO 8601**: `2024-01-15` (recommended)
- **US Format**: `01/15/2024`
- **EU Format**: `15.01.2024`

IRBStudio uses `pandas.to_datetime()` which handles most common formats automatically.

#### Credit Scores
- **Scale**: 0 to 1 (where 0 = best, 1 = worst)
- **Conversion**: If your scores are 300-850, convert using:
  ```python
  df['score_normalized'] = 1 - ((df['score'] - 300) / 550)
  ```

#### Rating Grades
- **Format**: String labels (e.g., 'AAA', 'A', 'BBB', 'B')
- **Case**: Case-sensitive (ensure consistency)
- **Coverage**: All ratings in data must be in `rating_pd_map`

### Sample Data Structures

#### Example 1: Freddie Mac Format

```csv
loan_seq_number,monthly_reporting_period,current_upb,current_loan_delinquency_status,loan_age,zero_balance_code,foreclosure_date
100000000001,01/01/2024,250000,0,36,,
100000000002,01/01/2024,180000,2,48,,
100000000003,01/01/2024,320000,0,24,,
```

**Mapping:**
```yaml
column_mapping:
  loan_id: loan_seq_number
  exposure: current_upb
  date: monthly_reporting_period
  default_flag: current_loan_delinquency_status  # Map: 0-2 = 0, 3+ = 1
```

#### Example 2: Internal Format

```csv
loan_number,eod_date,outstanding_balance,pd_estimate,risk_score,risk_grade,is_default,new_default,ltv_ratio
L2024001,2024-01-31,275000,0.025,0.08,A,0,0,0.75
L2024002,2024-01-31,195000,0.055,0.14,B,0,0,0.82
```

**Mapping:**
```yaml
column_mapping:
  loan_id: loan_number
  exposure: outstanding_balance
  pd: pd_estimate
  score: risk_score
  rating: risk_grade
  date: eod_date
  default_flag: is_default
  into_default_flag: new_default
  ltv: ltv_ratio
```

### Data Quality Checks

Before running IRBStudio, validate your data:

```python
import pandas as pd
from irbstudio.data.loader import load_portfolio, load_config

# Load data
config = load_config("config.yaml")
portfolio = load_portfolio("portfolio.csv", config.column_mapping)

# Check for missing values
print("Missing values:")
print(portfolio.isnull().sum())

# Check date range
print(f"\nDate range: {portfolio['date'].min()} to {portfolio['date'].max()}")

# Check rating coverage
ratings_in_data = set(portfolio['rating'].unique())
ratings_in_config = set(config.scenarios[0].rating_pd_map.keys())
missing = ratings_in_data - ratings_in_config
if missing:
    print(f"\nWarning: Ratings in data but not in config: {missing}")

# Check PD ranges
print(f"\nPD range: {portfolio['pd'].min():.4f} to {portfolio['pd'].max():.4f}")
print(f"Score range: {portfolio['score'].min():.4f} to {portfolio['score'].max():.4f}")
```

---

## Running Analyses

### Method 1: High-Level API (Recommended)

The simplest way to run analyses is using the high-level API functions:

#### Single Scenario Analysis

```python
from irbstudio import run_analysis

results = run_analysis(
    config_path="config.yaml",
    portfolio_path="portfolio.csv",
    n_iterations=1000,
    random_seed=42,
    output_dir="results",
    memory_efficient=False
)
```

#### Scenario Comparison

```python
from irbstudio import run_scenario_comparison

results = run_scenario_comparison(
    config_path="config.yaml",
    portfolio_path="portfolio.csv",
    n_iterations=1000,
    random_seed=42,
    output_dir="results"
)
```

### Method 2: Programmatic API (Advanced)

For more control, use the programmatic API:

```python
from irbstudio.data.loader import load_portfolio, load_config
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.engine.mortgage import AIRBMortgageCalculator, SAMortgageCalculator
from irbstudio.reporting.dashboard import create_dashboard

# 1. Load configuration and data
config = load_config("config.yaml")
portfolio_df = load_portfolio("portfolio.csv", config.column_mapping)

# 2. Create analysis engine
analysis = IntegratedAnalysis()

# 3. Add calculators
airb_calc = AIRBMortgageCalculator(
    regulatory_params={
        'lgd': 0.25,
        'asset_correlation': 0.15,
        'confidence_level': 0.999
    }
)
analysis.add_calculator('AIRB', airb_calc)

sa_calc = SAMortgageCalculator()
analysis.add_calculator('SA', sa_calc)

# 4. Create and add scenarios
for scenario_config in config.scenarios:
    simulator = PortfolioSimulator(
        portfolio_df=portfolio_df,
        score_to_rating_bounds={  # Define rating boundaries
            'A': (0.03, 0.10),
            'B': (0.10, 0.20)
        },
        rating_col='rating',
        loan_id_col='loan_id',
        date_col='reporting_date',
        default_col='default_flag',
        into_default_flag_col='into_default_flag',
        score_col='score',
        target_auc=scenario_config.pd_auc
    )
    
    analysis.add_scenario(
        scenario_name=scenario_config.name,
        simulator=simulator,
        n_iterations=1000
    )

# 5. Run scenarios
for scenario_name in analysis.scenarios.keys():
    results = analysis.run_scenario(
        scenario_name=scenario_name,
        random_seed=42,
        application_start_date='2024-01-01'
    )

# 6. Get results
summary = analysis.get_summary_stats('Baseline', 'AIRB')
percentiles = analysis.get_percentiles('Baseline', 'AIRB')

# 7. Generate dashboard
dashboard_html = create_dashboard(
    analysis_results=analysis.results,
    output_path="results/dashboard.html"
)
```

---

## Understanding Results

### Result Structure

Each scenario returns a nested dictionary:

```python
{
    'Baseline': {
        'AIRB': {
            'rwa_values': [array of RWA values],
            'mean': 5234567.89,
            'std': 123456.78,
            'median': 5198765.43,
            'percentiles': {
                'P5': 5098765.43,
                'P95': 5456789.01,
                'P99': 5567890.12
            }
        },
        'SA': { ... }
    },
    'Improved Model': { ... }
}
```

### Key Metrics

#### Mean RWA
- **Description**: Expected (average) RWA across all simulations
- **Use**: Primary metric for capital planning
- **Interpretation**: Higher mean = higher capital requirements

#### Standard Deviation
- **Description**: Measure of RWA volatility/uncertainty
- **Use**: Understanding model risk and uncertainty
- **Interpretation**: Higher std = more uncertain outcomes

#### Percentiles
- **P5**: 5th percentile (optimistic scenario)
- **P50** (Median): Middle value (robust to outliers)
- **P95**: 95th percentile (conservative scenario)
- **P99**: 99th percentile (stress scenario)

**Use Cases:**
- **Stress Testing**: Use P95 or P99 for stress capital requirements
- **Planning**: Use P50 (median) for robust planning
- **Risk Appetite**: Compare P95 to maximum acceptable RWA

### Capital Impact Analysis

#### Capital Delta
- **Formula**: `Capital Delta = (Baseline Mean RWA - Improved Mean RWA) * 0.08`
- **Interpretation**: Estimated capital savings from model improvement
- **Example**: If delta = $1M, bank saves $1M in regulatory capital

#### Percentage Reduction
- **Formula**: `% Reduction = (Delta / Baseline Mean RWA) * 100`
- **Example**: 10% reduction means 10% less capital tied up

---

## Advanced Usage

### Custom Rating Bounds

Define explicit rating boundaries for more control:

```python
score_to_rating_bounds = {
    'AAA': (0.00, 0.02),
    'AA': (0.02, 0.05),
    'A': (0.05, 0.10),
    'BBB': (0.10, 0.20),
    'BB': (0.20, 0.30),
    'B': (0.30, 0.50),
    'CCC': (0.50, 1.00)
}

simulator = PortfolioSimulator(
    portfolio_df=portfolio_df,
    score_to_rating_bounds=score_to_rating_bounds,
    ...
)
```

### Memory-Efficient Mode

For large portfolios, enable memory-efficient mode:

```python
results = analysis.run_scenario(
    scenario_name='Baseline',
    random_seed=42,
    memory_efficient=True  # Discards intermediate results
)
```

**Trade-offs:**
- ✅ Lower memory usage
- ✅ Faster execution for large portfolios
- ❌ Cannot access full portfolio simulations
- ❌ Only summary statistics available

### Process All Dates

Simulate each date separately (advanced):

```python
results = analysis.run_scenario(
    scenario_name='Baseline',
    random_seed=42,
    process_all_dates=True  # Separate simulation per date
)
```

**Use Cases:**
- Time series analysis
- Understanding temporal dynamics
- Date-specific insights

### Custom RWA Breakdowns

Analyze RWA by segments:

```python
# By rating
rating_breakdown = analysis.summarize_rwa(
    scenario_name='Baseline',
    calculator_name='AIRB',
    breakdown='rating'
)

# By region (if available in data)
region_breakdown = analysis.summarize_rwa(
    scenario_name='Baseline',
    calculator_name='AIRB',
    breakdown='region'
)

# By product
product_breakdown = analysis.summarize_rwa(
    scenario_name='Baseline',
    calculator_name='AIRB',
    breakdown='product'
)
```

---

## Real-World Examples

### Example 1: Model Improvement Business Case

**Scenario**: Your bank wants to improve the PD model. What's the capital impact?

```python
from irbstudio import run_scenario_comparison

# Define scenarios
config = {
    'scenarios': [
        {
            'name': 'Current Model',
            'pd_auc': 0.72,  # Current AUC
            'portfolio_default_rate': 0.035,
            'lgd': 0.25,
            'new_loan_rate': 0.12
        },
        {
            'name': 'Improved Model',
            'pd_auc': 0.82,  # Target AUC
            'portfolio_default_rate': 0.035,
            'lgd': 0.25,
            'new_loan_rate': 0.12
        }
    ]
}

# Run comparison
results = run_scenario_comparison(
    config_path="config.yaml",
    portfolio_path="portfolio.csv",
    n_iterations=5000,  # Higher for business case
    random_seed=42
)

# Calculate capital savings
current_rwa = results['Current Model']['AIRB']['mean']
improved_rwa = results['Improved Model']['AIRB']['mean']
capital_savings = (current_rwa - improved_rwa) * 0.08

print(f"Current RWA: ${current_rwa:,.0f}")
print(f"Improved RWA: ${improved_rwa:,.0f}")
print(f"Capital Savings: ${capital_savings:,.0f}")
print(f"ROI: {(capital_savings / model_development_cost * 100):.1f}%")
```

### Example 2: Stress Testing

**Scenario**: Analyze capital under stressed conditions

```python
# Define stress scenarios
scenarios = [
    {'name': 'Baseline', 'pd_auc': 0.75, 'portfolio_default_rate': 0.03, 'lgd': 0.25},
    {'name': 'Mild Stress', 'pd_auc': 0.72, 'portfolio_default_rate': 0.05, 'lgd': 0.30},
    {'name': 'Severe Stress', 'pd_auc': 0.65, 'portfolio_default_rate': 0.08, 'lgd': 0.40}
]

# Run analysis
results = run_scenario_comparison(...)

# Analyze stress impact
for scenario in ['Baseline', 'Mild Stress', 'Severe Stress']:
    p95 = results[scenario]['AIRB']['percentiles']['P95']
    print(f"{scenario} P95 RWA: ${p95:,.0f}")
```

### Example 3: AIRB vs SA Comparison

**Scenario**: Should your bank use AIRB or SA?

```python
from irbstudio.engine.mortgage import AIRBMortgageCalculator, SAMortgageCalculator

# Add both calculators
analysis.add_calculator('AIRB', AIRBMortgageCalculator(...))
analysis.add_calculator('SA', SAMortgageCalculator())

# Run scenario
results = analysis.run_scenario('Baseline', random_seed=42)

# Compare
airb_mean = results['AIRB']['mean']
sa_mean = results['SA']['mean']
difference = sa_mean - airb_mean
percent_diff = (difference / sa_mean) * 100

print(f"AIRB RWA: ${airb_mean:,.0f}")
print(f"SA RWA: ${sa_mean:,.0f}")
print(f"Capital Savings with AIRB: ${difference * 0.08:,.0f}")
print(f"AIRB reduces RWA by {percent_diff:.1f}%")
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "Historical data is empty"

**Error:**
```
ValueError: Historical data is empty. Cannot proceed with simulation.
```

**Cause**: `application_start_date` is same as or before all portfolio dates.

**Solution**: Set `application_start_date` to split the portfolio:
```python
results = analysis.run_scenario(
    scenario_name='Baseline',
    application_start_date='2024-01-01'  # After some historical dates
)
```

#### Issue 2: "Rating X not found in rating_pd_map"

**Error:**
```
KeyError: 'Rating BB+ not found in rating_pd_map'
```

**Cause**: Portfolio contains rating grades not in configuration.

**Solution**: Add missing ratings to config:
```yaml
rating_pd_map:
  BB+: 0.045
  BB: 0.05
  BB-: 0.055
```

#### Issue 3: "Unsupported file type"

**Error:**
```
ValueError: Unsupported file type. Only CSV and Parquet formats are supported.
```

**Cause**: Trying to load Excel, compressed, or other formats.

**Solution**: Convert to CSV or Parquet:
```python
# Excel to CSV
import pandas as pd
df = pd.read_excel("portfolio.xlsx")
df.to_csv("portfolio.csv", index=False)
```

#### Issue 4: Beta Fitting Convergence Error

**Error:**
```
FitSolverError: Solver for the MLE equations failed to converge
```

**Cause**: Insufficient data or extreme parameter values.

**Solutions:**
1. Increase portfolio size (need 30+ loans minimum)
2. Ensure default rate is reasonable (1-10%)
3. Check for data quality issues
4. Add more historical dates

---

## Best Practices

### 1. Data Preparation
- ✅ Clean data before running IRBStudio
- ✅ Validate date formats and ranges
- ✅ Check for missing values
- ✅ Ensure rating consistency
- ✅ Use representative portfolio samples

### 2. Configuration
- ✅ Start with simple scenarios, add complexity gradually
- ✅ Document scenario assumptions
- ✅ Use consistent rating schemes across scenarios
- ✅ Validate regulatory parameters against guidelines

### 3. Simulation Settings
- ✅ Use 1,000 iterations for development
- ✅ Use 5,000+ iterations for final analysis
- ✅ Always set `random_seed` for reproducibility
- ✅ Enable `memory_efficient` for large portfolios

### 4. Analysis Workflow
- ✅ Run baseline scenario first to validate setup
- ✅ Compare AIRB vs SA before complex scenarios
- ✅ Analyze percentiles, not just means
- ✅ Document assumptions and results

### 5. Reporting
- ✅ Generate dashboards for stakeholder communication
- ✅ Export results to CSV for further analysis
- ✅ Include confidence intervals in reports
- ✅ Explain methodology and limitations

### 6. Version Control
- ✅ Track configuration files in version control
- ✅ Document changes to scenarios
- ✅ Archive results with timestamps
- ✅ Maintain reproducible analysis notebooks

### 7. Performance Optimization
- ✅ Use memory-efficient mode for >100K loans
- ✅ Filter unnecessary columns before loading
- ✅ Consider parallel processing for multiple scenarios
- ✅ Cache intermediate results when appropriate

---

## Next Steps

- **Explore Examples**: Check the `examples/` directory for more use cases
- **Read API Reference**: See `docs/api_reference.md` for detailed API documentation
- **Join Community**: (Coming soon) Connect with other IRBStudio users
- **Contribute**: See `CONTRIBUTING.md` for contribution guidelines

---

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/jacekkrawiec/IRBStudio/issues)
- **Documentation**: [Project Plan](project_plan.md) | [API Reference](api_reference.md)
- **Email**: Contact the maintainer for support

---

*Last Updated: January 2025*
