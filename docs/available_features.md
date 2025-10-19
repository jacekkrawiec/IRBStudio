# IRBStudio - Available Features

**Version:** 0.1.0  
**Last Updated:** October 19, 2025

This document provides a comprehensive overview of all features available in IRBStudio, organized by functionality area.

---

## 📋 Table of Contents

1. [High-Level API](#high-level-api)
2. [Data Management](#data-management)
3. [Configuration System](#configuration-system)
4. [Monte Carlo Simulation](#monte-carlo-simulation)
5. [RWA Calculators](#rwa-calculators)
6. [Scenario Analysis](#scenario-analysis)
7. [Reporting & Visualization](#reporting--visualization)
8. [Advanced Features](#advanced-features)
9. [Utility Functions](#utility-functions)

---

## 1. High-Level API

### 1.1 Complete Analysis Pipeline

**`run_analysis()`** - One-line portfolio analysis
```python
from irbstudio import run_analysis

results = run_analysis(
    config_path='config.yaml',
    portfolio_path='portfolio.csv',
    n_iterations=1000
)
```

**Features:**
- ✅ Automated workflow from config to results
- ✅ Multi-scenario support
- ✅ Multiple calculator execution (AIRB, SA)
- ✅ Progress tracking with callbacks
- ✅ Memory-efficient processing
- ✅ Automatic result export (CSV, HTML)
- ✅ Statistical summaries and comparisons

**Parameters:**
- `config_path` - YAML configuration file path
- `portfolio_path` - Portfolio data (CSV, Parquet)
- `output_dir` - Results directory (optional)
- `calculators` - Calculator types ['AIRB', 'SA']
- `n_iterations` - Monte Carlo iterations (default: 100)
- `memory_efficient` - Process one iteration at a time (default: True)
- `store_full_portfolio` - Keep complete DataFrames (default: False)
- `random_seed` - Reproducibility seed
- `progress_callback` - Custom progress function

**Returns:**
- `summary` - Statistical summary per scenario/calculator
- `comparisons` - Scenario comparison metrics
- `calculator_results` - Raw RWA results per iteration
- `execution_time` - Performance metrics

### 1.2 Scenario Comparison

**`run_scenario_comparison()`** - Compare two scenarios directly
```python
from irbstudio import run_scenario_comparison

comparison = run_scenario_comparison(
    baseline_config='baseline.yaml',
    alternative_config='improved.yaml',
    portfolio_path='portfolio.csv'
)
```

**Features:**
- ✅ Side-by-side scenario comparison
- ✅ Capital delta calculation
- ✅ Statistical significance testing
- ✅ Visualization generation

---

## 2. Data Management

### 2.1 Portfolio Data Loading

**`load_portfolio()`** - Load and validate portfolio data
```python
from irbstudio import load_portfolio

portfolio_df = load_portfolio(
    file_path='portfolio.csv',
    column_mapping={
        'loan_id': 'Loan_ID',
        'exposure': 'Outstanding_Balance'
    }
)
```

**Supported Formats:**
- ✅ CSV files (`.csv`)
- ✅ Parquet files (`.parquet`)
- ✅ Excel files (`.xlsx`, `.xls`)
- ✅ Compressed files (`.gz`, `.zip`)

**Features:**
- ✅ Automatic data type inference
- ✅ Date parsing
- ✅ Column name mapping
- ✅ Missing value handling
- ✅ Data quality validation
- ✅ Memory-efficient chunked reading

**Required Columns:**
- `loan_id` - Unique loan identifier
- `exposure` - Exposure at default (EAD)
- `pd` or `score` - Probability of default or credit score
- `rating` - Credit rating grade
- `reporting_date` - Observation date
- `default_flag` - Current default status
- `into_default_flag` - Default transition indicator

**Optional Columns:**
- `lgd` - Loss given default (exposure-level)
- `maturity` - Loan maturity
- `property_value` - Collateral value
- `ltv` - Loan-to-value ratio
- `segment` - Portfolio segment
- `product_type` - Product classification

### 2.2 Configuration Loading

**`load_config()`** - Load and validate YAML configuration
```python
from irbstudio import load_config

config = load_config('config.yaml')
```

**Features:**
- ✅ YAML parsing with validation
- ✅ Pydantic schema validation
- ✅ Default value handling
- ✅ Type checking
- ✅ Nested configuration support

---

## 3. Configuration System

### 3.1 Configuration Schema

**`Config`** - Main configuration object

**Sections:**
1. **Portfolio Settings** - Data source and column mappings
2. **Scenarios** - Model performance assumptions
3. **Regulatory Parameters** - AIRB/SA calculation settings
4. **Simulation Settings** - Monte Carlo configuration
5. **Output Settings** - Result export options

### 3.2 Scenario Definition

**`Scenario`** - Individual scenario configuration

**Parameters:**
- `name` - Scenario identifier
- `description` - Human-readable description
- `target_auc` - PD model discrimination (0.5-1.0)
- `asset_correlation` - Systemic risk factor (0-1)
- `bad_proportion` - Default rate assumption
- `application_start_date` - Cut-off for historical/application split

**Example:**
```yaml
scenarios:
  - name: "Baseline"
    description: "Current model performance"
    target_auc: 0.80
    asset_correlation: 0.15
    bad_proportion: 0.03
    
  - name: "Improved"
    description: "Enhanced PD model"
    target_auc: 0.90
    asset_correlation: 0.15
    bad_proportion: 0.03
```

### 3.3 Column Mapping

**`ColumnMapping`** - Map portfolio columns to IRBStudio schema

**Features:**
- ✅ Flexible column naming
- ✅ Support for different data sources
- ✅ Validation of required fields
- ✅ Default mappings for common formats

**Example:**
```yaml
column_mapping:
  loan_id: "Loan_Sequence_Number"
  exposure: "Current_Actual_UPB"
  pd: "simulated_pd"
  rating: "rating"
  date: "reporting_date"
```

### 3.4 Regulatory Parameters

**`RegulatoryParams`** - AIRB and SA calculation settings

**AIRB Parameters:**
- `asset_correlation` - ρ (rho) parameter (default: 0.15 for mortgages)
- `confidence_level` - VaR confidence level (default: 0.999)
- `lgd` - Loss given default (default: 0.25)
- `maturity_adjustment` - Apply maturity factor (default: False)

**SA Parameters:**
- `secured_portion_rw` - Risk weight for secured portion (default: 0.20)
- `unsecured_portion_rw` - Risk weight for unsecured portion (default: 0.75)
- `property_value_threshold` - LTV threshold for secured/unsecured split (default: 0.55)

---

## 4. Monte Carlo Simulation

### 4.1 Portfolio Simulator

**`PortfolioSimulator`** - Core simulation engine

**Features:**
- ✅ **Hybrid Simulation Approach**
  - Existing clients: Migration matrix-based transitions
  - New clients: Calibrated score generation
- ✅ **Procedurally Faithful** - Respects historical data patterns
- ✅ **AUC Calibration** - Target specific model discrimination
- ✅ **Merton Model** - Asset correlation framework
- ✅ **Beta Mixture Modeling** - Dual-component score distribution

**Key Methods:**

**`prepare_simulation()`** - Initialize simulation components
- Portfolio segmentation (historical vs. application)
- Client classification (existing vs. new)
- Distribution fitting (non-default vs. default)
- Migration matrix calculation
- Long-term PD estimation

**`simulate_once()`** - Single Monte Carlo iteration
- Systemic factor generation
- Historical rating simulation
- Migration matrix calculation
- New client score generation
- Existing client migration
- PD assignment

**`run_monte_carlo()`** - Multiple iteration execution
- Parallel iteration support
- Memory-efficient mode
- Random seed management
- Progress tracking

### 4.2 Score Generation

**Beta Mixture Model** - Dual-component distribution

**Features:**
- ✅ Supervised fitting (separate default/non-default)
- ✅ Unsupervised fitting (EM algorithm)
- ✅ AUC calibration via gamma parameter
- ✅ Boundary score handling
- ✅ Component weight estimation

**Calibration:**
```python
simulator = PortfolioSimulator(
    portfolio_df=data,
    target_auc=0.85,  # Calibrate to 85% AUC
    asset_correlation=0.15,
    random_seed=42
)
```

### 4.3 Migration Matrices

**Features:**
- ✅ Historical transition rate calculation
- ✅ Rating grade migration patterns
- ✅ Default transition modeling
- ✅ Stable state analysis

**Calculation Methods:**
- Observed historical migrations
- Simulated score-based migrations
- Validation against historical patterns

### 4.4 Segmentation

**Portfolio Segments:**
1. **Historical** - Training data (before application_start_date)
2. **Application** - Simulation target (after application_start_date)
3. **Existing Clients** - Appear in both historical and application
4. **New Clients** - Only in application period
5. **Defaulted** - Current defaults (fixed rating)

---

## 5. RWA Calculators

### 5.1 AIRB Calculator

**`AIRBMortgageCalculator`** - Advanced Internal Ratings-Based approach

**Features:**
- ✅ Basel III AIRB formula implementation
- ✅ Mortgage-specific calibration
- ✅ Maturity adjustment (optional)
- ✅ Capital multiplier (1.06)
- ✅ 12.5x scaling factor
- ✅ Exposure-level LGD support

**Formula Components:**
1. **Correlation Function**: ρ = 0.15 (fixed for mortgages)
2. **Maturity Adjustment**: b(PD) function (optional)
3. **Capital Requirement**: K(PD, LGD, ρ)
4. **Risk Weight**: RW = K × 12.5 × 1.06
5. **RWA**: RW × Exposure

**Parameters:**
```python
airb = AIRBMortgageCalculator({
    'asset_correlation': 0.15,      # ρ parameter
    'confidence_level': 0.999,      # 99.9% VaR
    'lgd': 0.25,                    # Default LGD
    'maturity_adjustment': False    # Disable maturity factor
})
```

**Methods:**
- `calculate_rw()` - Calculate risk weights
- `calculate_rwa()` - Calculate risk-weighted assets
- `calculate()` - Complete calculation with summary
- `summarize_rwa()` - Generate statistical summary

### 5.2 SA Calculator

**`SAMortgageCalculator`** - Standardized Approach

**Features:**
- ✅ Basel III SA formula
- ✅ LTV-based risk weighting
- ✅ Secured vs. unsecured split
- ✅ Property value consideration

**Risk Weight Logic:**
```
If LTV ≤ threshold (55%):
    RW = secured_rw (20%)
Else:
    Secured RW = secured_rw × (threshold × property_value)
    Unsecured RW = unsecured_rw × (exposure - secured_portion)
    Total RW = (Secured RW + Unsecured RW) / exposure
```

**Parameters:**
```python
sa = SAMortgageCalculator({
    'secured_portion_rw': 0.20,         # 20% for secured
    'unsecured_portion_rw': 0.75,       # 75% for unsecured
    'property_value_threshold': 0.55    # 55% LTV threshold
})
```

### 5.3 RWA Results

**`RWAResult`** - Calculation output container

**Properties:**
- `total_rwa` - Total risk-weighted assets
- `total_exposure` - Total exposure
- `capital_requirement` - 8% of RWA
- `portfolio` - DataFrame with RW and RWA columns
- `summary` - Statistical metrics
- `metadata` - Calculator configuration
- `by_date` - Date-specific breakdown (if enabled)

**Methods:**
- `get_breakdown(by)` - Get breakdown by field (rating, date, segment)
- `has_breakdown(by)` - Check if breakdown available
- `get_available_breakdowns()` - List all breakdown dimensions

**Date Breakdown:**
```python
# Access date-specific RWA
for date, metrics in result.by_date.items():
    print(f"{date}: RWA = ${metrics['total_rwa']:,.0f}")
```

---

## 6. Scenario Analysis

### 6.1 Integrated Analysis

**`IntegratedAnalysis`** - Multi-scenario orchestration

**Features:**
- ✅ Multiple scenario management
- ✅ Multiple calculator execution
- ✅ Memory-efficient iteration processing
- ✅ Date-based RWA breakdown
- ✅ Statistical aggregation
- ✅ Scenario comparison

**Workflow:**
```python
analysis = IntegratedAnalysis(date_column='reporting_date')

# Add calculators
analysis.add_calculator('AIRB', airb_calculator)
analysis.add_calculator('SA', sa_calculator)

# Add scenarios
analysis.add_scenario('Baseline', baseline_simulator, n_iterations=1000)
analysis.add_scenario('Improved', improved_simulator, n_iterations=1000)

# Run analysis
results = analysis.run_scenario(
    'Baseline',
    calculator_names=['AIRB', 'SA'],
    memory_efficient=True,
    process_all_dates=True
)
```

### 6.2 Scenario Comparison

**Features:**
- ✅ Capital delta calculation (absolute and percentage)
- ✅ Statistical significance testing
- ✅ Percentile comparison
- ✅ Distribution overlap analysis

**Metrics:**
- Mean RWA difference
- Median RWA difference
- Standard deviation changes
- Percentile shifts (P5, P25, P50, P75, P95)
- Capital savings ($)
- Capital reduction (%)

### 6.3 Statistical Summaries

**`get_summary_stats()`** - Scenario statistics
- Mean, Median, Std Dev
- Min, Max
- Skewness, Kurtosis
- Coefficient of Variation

**`get_percentiles()`** - Percentile analysis
- Configurable percentiles (default: 5, 25, 50, 75, 95)
- VaR-style risk metrics
- Tail risk assessment

**`get_rwa_distribution()`** - Full distribution
- Pandas Series of all RWA values
- Suitable for custom analysis
- Histogram input

---

## 7. Reporting & Visualization

### 7.1 Interactive Visualizations

All visualization functions use **Plotly** for interactive HTML charts.

**`create_rwa_distribution_plot()`** - RWA distribution histogram
```python
from irbstudio.reporting import create_rwa_distribution_plot

fig = create_rwa_distribution_plot(
    results,
    scenario_name='Baseline',
    calculator_name='AIRB',
    show_stats=True
)
fig.write_html('distribution.html')
```

**Features:**
- ✅ Histogram with KDE overlay
- ✅ Mean, median, std dev annotations
- ✅ Percentile markers
- ✅ Sample size display
- ✅ Interactive hover tooltips

**`create_scenario_comparison_plot()`** - Multi-scenario comparison
```python
fig = create_scenario_comparison_plot(
    results,
    scenarios=['Baseline', 'Improved', 'Stressed'],
    calculator_name='AIRB'
)
```

**Features:**
- ✅ Overlaid distributions
- ✅ Color-coded scenarios
- ✅ Summary statistics table
- ✅ Capital delta annotations

**`create_waterfall_chart()`** - Scenario impact breakdown
```python
fig = create_waterfall_chart(
    scenario1='Baseline',
    scenario2='Improved',
    results=results,
    calculator_name='AIRB'
)
```

**Features:**
- ✅ Step-by-step impact visualization
- ✅ Absolute and percentage changes
- ✅ Component breakdown
- ✅ Net effect summary

**`create_summary_table()`** - Statistical comparison table
```python
fig = create_summary_table(
    results,
    calculator_names=['AIRB', 'SA']
)
```

**Features:**
- ✅ All scenarios and calculators
- ✅ Key statistics (mean, median, P5, P95)
- ✅ Sortable columns
- ✅ Export to CSV/Excel

**`create_percentile_plot()`** - Percentile comparison
```python
fig = create_percentile_plot(
    results,
    scenario_name='Baseline',
    calculator_name='AIRB',
    percentiles=[5, 25, 50, 75, 95]
)
```

**Features:**
- ✅ Bar chart of percentiles
- ✅ Risk metric visualization
- ✅ VaR-style display
- ✅ Confidence interval bands

### 7.2 Date-Based Visualizations

**`create_rwa_by_date_plot()`** - RWA evolution over time
```python
fig = create_rwa_by_date_plot(
    results_by_iteration,
    scenario_name='Baseline',
    calculator_name='AIRB'
)
```

**Features:**
- ✅ Time series for each iteration
- ✅ Mean line with confidence intervals
- ✅ P5-P95 shaded region (90% CI)
- ✅ Interactive date selection
- ✅ Temporal pattern analysis

**`create_rwa_distribution_by_date_plot()`** - Date-specific distribution
```python
fig = create_rwa_distribution_by_date_plot(
    results_by_iteration,
    scenario_name='Baseline',
    calculator_name='AIRB',
    target_date='2024-12-31'  # Optional, defaults to last date
)
```

**Features:**
- ✅ Histogram for specific reporting date
- ✅ Comparison across iterations for same date
- ✅ Statistical annotations
- ✅ Useful for period-end analysis

### 7.3 Dashboard Generation

**Comprehensive HTML Dashboard**
- Automatic multi-panel layout
- Embedded interactive charts
- Navigation menu
- Summary statistics table
- Scenario comparison section
- Export functionality

---

## 8. Advanced Features

### 8.1 Memory-Efficient Processing

**`memory_efficient=True`** - Process iterations one at a time

**Features:**
- ✅ Reduces memory footprint by ~90%
- ✅ Handles portfolios with 10M+ rows
- ✅ Automatic garbage collection
- ✅ Progress tracking
- ✅ No intermediate storage

**Use Cases:**
- Large portfolios (millions of loans)
- Limited RAM environments
- High iteration counts (1000+)
- Long time series (many dates)

### 8.2 Date Breakdown

**`process_all_dates=True`** - Calculate RWA by reporting date

**Features:**
- ✅ Temporal RWA analysis
- ✅ Date-specific capital requirements
- ✅ Trend identification
- ✅ Seasonal pattern detection
- ✅ Period-end reporting

**Access Date Breakdown:**
```python
result = analysis.results['Baseline']['calculator_results']['AIRB']['results'][0]

# Option 1: Direct property access
for date, metrics in result.by_date.items():
    print(f"{date}: RWA = ${metrics['total_rwa']:,.0f}")

# Option 2: Using get_breakdown
date_breakdown = result.get_breakdown('date')
```

**Date Metrics:**
- `total_rwa` - RWA for specific date
- `total_exposure` - Exposure for specific date
- `average_risk_weight` - Mean RW for date
- `weighted_average_rw` - Exposure-weighted RW

### 8.3 Portfolio Filtering

**Custom Filter Functions**

```python
def high_ltv_filter(df):
    """Filter to high LTV loans only"""
    return df[df['ltv'] > 0.80]

results = analysis.run_scenario(
    'Baseline',
    portfolio_filter=high_ltv_filter
)
```

**Use Cases:**
- Segment analysis (retail vs. wholesale)
- Geographic concentration (by state/region)
- Product type analysis (fixed vs. variable)
- Risk concentration (high LTV, low FICO)
- Vintage analysis (by origination year)

### 8.4 Reproducibility

**Random Seed Control**
```python
results = run_analysis(
    config_path='config.yaml',
    portfolio_path='portfolio.csv',
    random_seed=42  # Reproducible results
)
```

**Features:**
- ✅ Deterministic simulations
- ✅ Iteration-specific seeds (base_seed + iteration)
- ✅ Cross-validation support
- ✅ Model validation

### 8.5 Progress Tracking

**Custom Progress Callbacks**
```python
def my_progress_callback(step, progress):
    """Custom progress handler"""
    print(f"[{progress:.1%}] {step}")

results = run_analysis(
    config_path='config.yaml',
    portfolio_path='portfolio.csv',
    progress_callback=my_progress_callback
)
```

**Progress Steps:**
1. Loading configuration
2. Loading portfolio data
3. Preparing simulators
4. Running Monte Carlo iterations
5. Calculating RWA
6. Generating summaries
7. Exporting results

---

## 9. Utility Functions

### 9.1 Logging

**`get_logger()`** - Structured logging

**Features:**
- ✅ Configurable log levels
- ✅ Component-specific loggers
- ✅ Timestamp and module tracking
- ✅ File and console output

**Usage:**
```python
from irbstudio.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Analysis started")
```

### 9.2 Data Validation

**Automatic Validation:**
- Required columns present
- Data types correct
- Date parsing successful
- No critical missing values
- Numeric ranges valid

**Manual Validation:**
```python
from irbstudio.data.loader import validate_portfolio

is_valid, errors = validate_portfolio(portfolio_df)
if not is_valid:
    for error in errors:
        print(f"Validation error: {error}")
```

### 9.3 Column Mapping

**Flexible Column Names:**
```python
column_mapping = {
    'loan_id': 'LOAN_ID',
    'exposure': 'BALANCE',
    'pd': 'PD_VALUE',
    'rating': 'RATING_CODE',
    'date': 'RPT_DATE'
}
```

**Supports:**
- Different naming conventions
- Multiple data sources
- Legacy system integration
- Custom field names

---

## 🎯 Feature Matrix

| Feature | Status | API Level | Complexity |
|---------|--------|-----------|------------|
| **Core Functionality** |
| Monte Carlo Simulation | ✅ Complete | High/Low | Medium |
| AIRB Calculator | ✅ Complete | High/Low | Low |
| SA Calculator | ✅ Complete | High/Low | Low |
| Multi-Scenario Analysis | ✅ Complete | High/Low | Medium |
| Configuration System | ✅ Complete | High | Low |
| **Data Management** |
| CSV/Parquet Loading | ✅ Complete | High/Low | Low |
| Column Mapping | ✅ Complete | High/Low | Low |
| Data Validation | ✅ Complete | High/Low | Low |
| Memory-Efficient Mode | ✅ Complete | High/Low | Medium |
| **Analysis Features** |
| Statistical Summaries | ✅ Complete | High/Low | Low |
| Percentile Analysis | ✅ Complete | High/Low | Low |
| Scenario Comparison | ✅ Complete | High | Medium |
| Date Breakdown | ✅ Complete | Low | Medium |
| Portfolio Filtering | ✅ Complete | Low | Low |
| **Visualization** |
| Distribution Plots | ✅ Complete | Low | Low |
| Waterfall Charts | ✅ Complete | Low | Low |
| Summary Tables | ✅ Complete | Low | Low |
| Percentile Plots | ✅ Complete | Low | Low |
| Date-Based Charts | ✅ Complete | Low | Medium |
| Comprehensive Dashboard | ✅ Complete | High | Medium |
| **Advanced** |
| AUC Calibration | ✅ Complete | Low | High |
| Beta Mixture Model | ✅ Complete | Low | High |
| Migration Matrices | ✅ Complete | Low | Medium |
| Merton Framework | ✅ Complete | Low | High |
| Progress Callbacks | ✅ Complete | High | Low |
| Reproducibility | ✅ Complete | High/Low | Low |

**Legend:**
- **API Level:** High = Simple user API, Low = Advanced/developer API
- **Complexity:** User difficulty level

---

## 📚 Related Documentation

- [Quick Start Guide](quick_start.md) - Get started in 5 minutes
- [API Reference](api_reference.md) - Complete API documentation
- [Configuration Guide](configuration.md) - YAML config details
- [Scenario Design Guide](scenario_design.md) - Creating effective scenarios
- [Methodology](methodology.md) - Technical approach and formulas
- [Examples](../examples/) - Working code examples

---

## 🔄 Version History

### v0.1.0 (Current)
- ✅ Complete AIRB and SA implementation
- ✅ Monte Carlo simulation engine
- ✅ Multi-scenario analysis
- ✅ Interactive visualizations
- ✅ Date breakdown functionality
- ✅ Memory-efficient processing
- ✅ Comprehensive documentation

### Roadmap (Future)
- 🔜 LGD simulation (beyond fixed values)
- 🔜 EAD simulation
- 🔜 Additional asset classes (corporate, retail)
- 🔜 Model monitoring features
- 🔜 Backtesting functionality
- 🔜 Web-based interface
- 🔜 Cloud deployment support

---

## 💡 Feature Requests

Have a feature request? Please open an issue on GitHub with the label `enhancement`.

**Popular Requests:**
1. Stress testing scenarios
2. Regulatory reporting templates
3. Model validation tools
4. Real-time monitoring
5. API endpoints for automation

---

**Last Updated:** October 19, 2025  
**Maintained By:** IRBStudio Team  
**License:** MIT
