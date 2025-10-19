# IRBStudio Tutorial: Analyzing Freddie Mac Mortgage Data

This tutorial demonstrates how to use IRBStudio with real-world mortgage portfolio data, specifically using the structure of Freddie Mac's Single-Family Loan-Level Dataset.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Freddie Mac Data](#understanding-freddie-mac-data)
3. [Data Preparation](#data-preparation)
4. [Configuration Setup](#configuration-setup)
5. [Running the Analysis](#running-the-analysis)
6. [Interpreting Results](#interpreting-results)
7. [Advanced Examples](#advanced-examples)

---

## Introduction

### About Freddie Mac Data

Freddie Mac publishes anonymized Single-Family Loan-Level Dataset that includes:
- **Origination Data**: Loan characteristics at origination
- **Performance Data**: Monthly loan performance updates

This tutorial shows how to:
1. Prepare Freddie Mac data for IRBStudio
2. Create appropriate configurations
3. Run AIRB scenario analysis
4. Interpret capital impact results

### Dataset Source

**Freddie Mac Single-Family Loan-Level Dataset**
- Source: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
- Format: Pipe-delimited text files (origination + performance)
- Size: Millions of loans, decades of performance
- Cost: Free (registration required)

---

## Understanding Freddie Mac Data

### File Structure

Fannie Mae provides two types of files:

**1. Origination Files** (`sample_orig_YYYY.txt`)
Contains loan characteristics at origination:
- Credit score
- LTV ratio
- DTI ratio
- Loan purpose
- Property type
- Number of borrowers
- First-time homebuyer flag

**2. Servicing Files** (`sample_svcg_YYYY.txt`)
Contains monthly performance updates:
- Current UPB (unpaid principal balance)
- Delinquency status
- Loan age
- Months to maturity
- Modification flag
- Zero balance code (payoff/default)
- Foreclosure date

### Key Fields Mapping

| IRBStudio Field | Fannie Mae Field | File | Description |
|-----------------|------------------|------|-------------|
| loan_id | LOAN_SEQUENCE_NUMBER | Both | Unique identifier |
| exposure | CURRENT_ACTUAL_UPB | Servicing | Current balance |
| date | MONTHLY_REPORTING_PERIOD | Servicing | Reporting date |
| default_flag | CURRENT_LOAN_DELINQUENCY_STATUS | Servicing | 0-2 = current, 3+ = default |
| ltv | ORIGINAL_LTV | Origination | Loan-to-value at origination |
| score | CREDIT_SCORE | Origination | FICO score |

### Challenge: No Explicit PD or Rating

Fannie Mae data doesn't include:
- ❌ PD (Probability of Default) values
- ❌ Internal rating grades
- ❌ Credit scores in 0-1 scale

**Solution**: We'll derive these from available data:
1. **PD**: Estimate from delinquency status and loan characteristics
2. **Rating**: Bin loans by credit score
3. **Score**: Normalize FICO (300-850) to 0-1 scale

---

## Data Preparation

### Step 1: Load Raw Fannie Mae Data

First, let's load the raw data files:

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Fannie Mae origination file columns (sample)
orig_columns = [
    'loan_sequence_number', 'credit_score', 'first_payment_date',
    'first_time_homebuyer_flag', 'maturity_date', 'msa',
    'mi_percentage', 'number_of_units', 'occupancy_status',
    'original_cltv', 'original_dti', 'original_upb',
    'original_ltv', 'original_interest_rate', 'channel',
    'prepayment_penalty_flag', 'product_type', 'property_state',
    'property_type', 'postal_code', 'loan_sequence_number_dup',
    'loan_purpose', 'original_loan_term', 'number_of_borrowers',
    'seller_name', 'servicer_name', 'super_conforming_flag'
]

# Load origination data
orig_df = pd.read_csv(
    'data/FM/sample_orig_2024.txt',
    sep='|',
    names=orig_columns,
    header=None
)

# Servicing file columns (sample)
svcg_columns = [
    'loan_sequence_number', 'monthly_reporting_period',
    'current_actual_upb', 'current_loan_delinquency_status',
    'loan_age', 'remaining_months_to_maturity',
    'repurchase_flag', 'modification_flag', 'zero_balance_code',
    'zero_balance_effective_date', 'current_interest_rate',
    'current_deferred_upb', 'due_date_of_last_paid_installment',
    'mi_recoveries', 'net_sales_proceeds', 'non_mi_recoveries',
    'expenses', 'legal_costs', 'maintenance_costs',
    'taxes_and_insurance', 'miscellaneous_expenses',
    'actual_loss_calculation', 'modification_cost'
]

# Load servicing data
svcg_df = pd.read_csv(
    'data/FM/sample_svcg_2024.txt',
    sep='|',
    names=svcg_columns,
    header=None
)

print(f"Origination records: {len(orig_df):,}")
print(f"Servicing records: {len(svcg_df):,}")
```

### Step 2: Merge and Transform Data

```python
# Merge origination and servicing data
portfolio = svcg_df.merge(
    orig_df,
    on='loan_sequence_number',
    how='inner'
)

# Parse dates
portfolio['monthly_reporting_period'] = pd.to_datetime(
    portfolio['monthly_reporting_period'],
    format='%m/%d/%Y'
)

# Create derived fields
portfolio['normalized_score'] = 1 - (
    (portfolio['credit_score'] - 300) / 550
)

# Map delinquency to default flag
# 0 = current, 1 = 30 days, 2 = 60 days, 3+ = default
portfolio['default_flag'] = (
    portfolio['current_loan_delinquency_status'] >= 3
).astype(int)

# Create into_default_flag (new defaults this period)
portfolio = portfolio.sort_values(['loan_sequence_number', 'monthly_reporting_period'])
portfolio['prev_default'] = portfolio.groupby('loan_sequence_number')['default_flag'].shift(1).fillna(0)
portfolio['into_default_flag'] = (
    (portfolio['default_flag'] == 1) & (portfolio['prev_default'] == 0)
).astype(int)

# Create rating grades based on credit score
def assign_rating(score):
    if score >= 780: return 'AAA'
    elif score >= 740: return 'AA'
    elif score >= 700: return 'A'
    elif score >= 660: return 'BBB'
    elif score >= 620: return 'BB'
    elif score >= 580: return 'B'
    else: return 'CCC'

portfolio['rating'] = portfolio['credit_score'].apply(assign_rating)

# Estimate PD from delinquency and score
# Simple heuristic: use historical default rates by rating
rating_pd = {
    'AAA': 0.0005, 'AA': 0.001, 'A': 0.005,
    'BBB': 0.01, 'BB': 0.03, 'B': 0.05, 'CCC': 0.10
}
portfolio['pd'] = portfolio['rating'].map(rating_pd)

# Add some noise to PD based on individual characteristics
portfolio['pd'] = portfolio['pd'] * (
    1 + 0.2 * (portfolio['original_ltv'] - 80) / 20
)  # Adjust for LTV
portfolio['pd'] = portfolio['pd'].clip(0.0001, 0.50)  # Reasonable bounds

print("\nTransformed Portfolio Sample:")
print(portfolio[['loan_sequence_number', 'monthly_reporting_period', 
               'current_actual_upb', 'credit_score', 'rating', 
               'pd', 'normalized_score', 'default_flag']].head())
```

### Step 3: Prepare IRBStudio Format

```python
# Select relevant columns and rename
irbstudio_portfolio = portfolio[[
    'loan_sequence_number',
    'current_actual_upb',
    'pd',
    'normalized_score',
    'rating',
    'monthly_reporting_period',
    'default_flag',
    'into_default_flag',
    'original_ltv'
]].rename(columns={
    'loan_sequence_number': 'loan_id',
    'current_actual_upb': 'balance',
    'normalized_score': 'score',
    'monthly_reporting_period': 'reporting_date',
    'original_ltv': 'ltv'
})

# Filter to recent data (e.g., last 12 months)
cutoff_date = irbstudio_portfolio['reporting_date'].max() - pd.DateOffset(months=12)
irbstudio_portfolio = irbstudio_portfolio[
    irbstudio_portfolio['reporting_date'] >= cutoff_date
]

# Remove loans with missing critical data
irbstudio_portfolio = irbstudio_portfolio.dropna(
    subset=['balance', 'pd', 'score', 'rating']
)

# Save to CSV
irbstudio_portfolio.to_csv(
    'data/fannie_mae_portfolio_prepared.csv',
    index=False
)

print(f"\nPrepared portfolio: {len(irbstudio_portfolio):,} records")
print(f"Date range: {irbstudio_portfolio['reporting_date'].min()} to {irbstudio_portfolio['reporting_date'].max()}")
print(f"Total exposure: ${irbstudio_portfolio['balance'].sum():,.0f}")
```

---

## Configuration Setup

### Create Configuration for Fannie Mae Analysis

```yaml
# fannie_mae_config.yaml

# Map prepared data to IRBStudio canonical fields
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

# Regulatory parameters for US mortgage portfolio
regulatory:
  jurisdiction: us
  asset_correlation: 0.15  # Basel standard for mortgage
  confidence_level: 0.999  # 99.9% confidence

# Scenarios to analyze
scenarios:
  # Current State: Baseline with current model
  - name: "Current Model"
    description: "Current PD model performance (AUC ~0.72 typical for FICO-based)"
    pd_auc: 0.72
    portfolio_default_rate: 0.015  # 1.5% typical for Fannie Mae
    lgd: 0.25  # 25% LGD for first-lien mortgages
    new_loan_rate: 0.08  # 8% new originations monthly
    rating_pd_map:
      AAA: 0.0005
      AA: 0.001
      A: 0.005
      BBB: 0.01
      BB: 0.03
      B: 0.05
      CCC: 0.10

  # Improved Model: Better discrimination with additional data
  - name: "Enhanced Model"
    description: "Improved model with payment history, DTI, property data (AUC ~0.78)"
    pd_auc: 0.78
    portfolio_default_rate: 0.015
    lgd: 0.25
    new_loan_rate: 0.08
    rating_pd_map:
      AAA: 0.0005
      AA: 0.001
      A: 0.005
      BBB: 0.01
      BB: 0.03
      B: 0.05
      CCC: 0.10

  # Stress Scenario: Economic downturn
  - name: "Stress Scenario"
    description: "Recession scenario with increased defaults and lower discrimination"
    pd_auc: 0.68  # Model discrimination degrades in stress
    portfolio_default_rate: 0.04  # 4% default rate (stress)
    lgd: 0.35  # Higher LGD in stress (longer foreclosures, lower recoveries)
    new_loan_rate: 0.03  # Lower originations in stress
    rating_pd_map:
      AAA: 0.001
      AA: 0.002
      A: 0.01
      BBB: 0.02
      BB: 0.06
      B: 0.10
      CCC: 0.20
```

---

## Running the Analysis

### Example 1: Basic Scenario Comparison

```python
from irbstudio import run_scenario_comparison

# Run comparison of current vs. enhanced model
results = run_scenario_comparison(
    config_path="fannie_mae_config.yaml",
    portfolio_path="data/fannie_mae_portfolio_prepared.csv",
    n_iterations=5000,  # Higher iterations for business case
    random_seed=42,
    output_dir="results/fannie_mae_analysis"
)

# Print summary
print("\n" + "="*60)
print("FANNIE MAE PORTFOLIO: CAPITAL IMPACT ANALYSIS")
print("="*60)

for scenario_name in ['Current Model', 'Enhanced Model', 'Stress Scenario']:
    if scenario_name in results:
        airb_stats = results[scenario_name]['AIRB']
        print(f"\n{scenario_name}:")
        print(f"  Mean RWA:    ${airb_stats['mean']:>15,.0f}")
        print(f"  Std Dev:     ${airb_stats['std']:>15,.0f}")
        print(f"  P95 RWA:     ${airb_stats['percentiles']['P95']:>15,.0f}")

# Capital savings from model improvement
if 'capital_delta' in results:
    print(f"\nCapital Savings (Current → Enhanced): ${results['capital_delta']:>15,.0f}")
    print(f"                                        (RWA reduction × 8% capital ratio)")

print("\n" + "="*60)
print(f"Dashboard: results/fannie_mae_analysis/scenario_comparison_dashboard.html")
print("="*60)
```

### Example 2: Detailed AIRB vs SA Comparison

```python
from irbstudio import load_config
from irbstudio.data.loader import load_portfolio
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.engine.mortgage import AIRBMortgageCalculator, SAMortgageCalculator

# Load data
config = load_config("fannie_mae_config.yaml")
portfolio_df = load_portfolio(
    "data/fannie_mae_portfolio_prepared.csv",
    config.column_mapping
)

# Create analysis engine
analysis = IntegratedAnalysis()

# Add both calculators
analysis.add_calculator('AIRB', AIRBMortgageCalculator(
    regulatory_params={
        'lgd': 0.25,
        'asset_correlation': 0.15,
        'confidence_level': 0.999
    }
))
analysis.add_calculator('SA', SAMortgageCalculator())

# Create simulator for current model
current_scenario = config.scenarios[0]  # "Current Model"
simulator = PortfolioSimulator(
    portfolio_df=portfolio_df,
    score_to_rating_bounds={
        'AAA': (0.00, 0.02),
        'AA': (0.02, 0.05),
        'A': (0.05, 0.10),
        'BBB': (0.10, 0.20),
        'BB': (0.20, 0.30),
        'B': (0.30, 0.50),
        'CCC': (0.50, 1.00)
    },
    rating_col='rating',
    loan_id_col='loan_id',
    date_col='reporting_date',
    default_col='default_flag',
    into_default_flag_col='into_default_flag',
    score_col='score',
    target_auc=current_scenario.pd_auc
)

# Add scenario
analysis.add_scenario('Current Model', simulator, n_iterations=2000)

# Run with both calculators
results = analysis.run_scenario(
    scenario_name='Current Model',
    random_seed=42,
    application_start_date='2024-01-01'
)

# Compare AIRB vs SA
airb_mean = results['AIRB']['mean']
sa_mean = results['SA']['mean']
savings = (sa_mean - airb_mean) * 0.08

print("\n" + "="*60)
print("AIRB vs STANDARDIZED APPROACH COMPARISON")
print("="*60)
print(f"Standardized Approach RWA:  ${sa_mean:>15,.0f}")
print(f"AIRB Approach RWA:          ${airb_mean:>15,.0f}")
print(f"RWA Reduction:              ${sa_mean - airb_mean:>15,.0f}")
print(f"Capital Savings:            ${savings:>15,.0f}")
print(f"Percentage Reduction:       {((sa_mean - airb_mean) / sa_mean * 100):>15.1f}%")
print("="*60)
```

---

## Interpreting Results

### Understanding Output Metrics

#### 1. Mean RWA
- **Definition**: Average RWA across all Monte Carlo simulations
- **Use**: Primary metric for capital planning and budgeting
- **Typical Values** (for $1B portfolio):
  - Conservative: $400M - $600M (40-60% RWA density)
  - Moderate: $250M - $400M (25-40%)
  - Aggressive: $150M - $250M (15-25%)

#### 2. Standard Deviation
- **Definition**: Measure of RWA volatility/uncertainty
- **Use**: Understanding model risk
- **Interpretation**:
  - Low (< 5% of mean): Stable, low uncertainty
  - Medium (5-15%): Typical for established portfolios
  - High (> 15%): High uncertainty, consider more data or model validation

#### 3. P95 RWA
- **Definition**: 95th percentile (only 5% chance of exceeding)
- **Use**: Stress testing and capital buffer determination
- **Regulatory**: Often used for ICAAP stress scenarios

### Dashboard Visualization Guide

The generated dashboard includes:

**1. Distribution Plots**
- **Histogram**: Shows RWA frequency distribution
- **KDE Overlay**: Smooth probability density
- **Percentile Lines**: P5, P50, P95 markers
- **Interpretation**: Look for skewness, multiple modes

**2. Scenario Comparison**
- **Bar Chart**: Side-by-side mean RWA comparison
- **Error Bars**: Show ±1 standard deviation
- **Use**: Quick visual comparison of scenarios

**3. Summary Statistics Table**
| Metric | Current Model | Enhanced Model | Stress |
|--------|---------------|----------------|--------|
| Mean | $325M | $298M | $487M |
| Std Dev | $18M | $16M | $34M |
| P95 | $356M | $325M | $545M |
| Skew | 0.15 | 0.12 | 0.28 |

**4. Waterfall Chart**
- Shows component-by-component RWA breakdown
- Useful for understanding drivers of change

### Business Insights

#### Model Improvement Business Case

```
Current Model:     Mean RWA = $325M
Enhanced Model:    Mean RWA = $298M
RWA Reduction:     $27M
Capital Savings:   $27M × 8% = $2.16M

If model development cost = $500K
ROI = ($2.16M - $0.5M) / $0.5M = 332%
Payback period = 0.3 years (~4 months)
```

#### AIRB vs SA Decision

```
SA Approach:       Mean RWA = $485M
AIRB Approach:     Mean RWA = $325M
RWA Reduction:     $160M
Capital Savings:   $160M × 8% = $12.8M

Annual savings justify AIRB implementation costs and ongoing model maintenance.
```

---

## Advanced Examples

### Example 3: Vintage Analysis

Analyze different loan vintages separately:

```python
# Group by origination year
portfolio_df['origination_year'] = portfolio_df['reporting_date'].dt.year - portfolio_df['loan_age'] // 12

vintages = {}
for year in [2020, 2021, 2022, 2023]:
    vintage_df = portfolio_df[portfolio_df['origination_year'] == year]
    
    simulator = PortfolioSimulator(
        portfolio_df=vintage_df,
        target_auc=0.75
    )
    
    analysis.add_scenario(f'Vintage_{year}', simulator, n_iterations=1000)
    results = analysis.run_scenario(f'Vintage_{year}', random_seed=42)
    
    vintages[year] = results['AIRB']['mean']

# Compare vintages
print("\nRWA by Vintage:")
for year, rwa in vintages.items():
    print(f"  {year}: ${rwa:,.0f}")
```

### Example 4: Geographic Segmentation

Analyze RWA by state/region:

```python
# Assume 'property_state' column exists
states = portfolio_df['property_state'].unique()

state_rwa = {}
for state in states[:5]:  # Top 5 states by volume
    state_df = portfolio_df[portfolio_df['property_state'] == state]
    
    if len(state_df) < 100:  # Skip small segments
        continue
    
    simulator = PortfolioSimulator(portfolio_df=state_df, target_auc=0.75)
    analysis.add_scenario(f'State_{state}', simulator, n_iterations=500)
    results = analysis.run_scenario(f'State_{state}', random_seed=42)
    
    state_rwa[state] = {
        'mean': results['AIRB']['mean'],
        'exposure': state_df['balance'].sum(),
        'rwa_density': results['AIRB']['mean'] / state_df['balance'].sum()
    }

# Print summary
print("\nRWA by State:")
print(f"{'State':<10} {'Exposure':>15} {'RWA':>15} {'Density':>10}")
print("-" * 55)
for state, metrics in state_rwa.items():
    print(f"{state:<10} ${metrics['exposure']:>14,.0f} ${metrics['mean']:>14,.0f} {metrics['rwa_density']:>9.1%}")
```

### Example 5: Temporal Dynamics

Analyze how RWA evolves over time:

```python
# Run with process_all_dates to see temporal dynamics
results = analysis.run_scenario(
    scenario_name='Current Model',
    random_seed=42,
    process_all_dates=True  # Separate simulation for each date
)

# Extract temporal results (if stored)
if 'date_results' in results:
    import plotly.graph_objects as go
    
    dates = sorted(results['date_results'].keys())
    mean_rwa = [results['date_results'][d]['AIRB']['mean'] for d in dates]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=mean_rwa,
        mode='lines+markers',
        name='Mean RWA'
    ))
    fig.update_layout(
        title='RWA Evolution Over Time',
        xaxis_title='Date',
        yaxis_title='Mean RWA ($)',
        hovermode='x unified'
    )
    fig.write_html('results/temporal_rwa.html')
```

---

## Summary and Next Steps

### Key Takeaways

1. **Data Preparation**: Fannie Mae data requires transformation to IRBStudio format
2. **Rating Creation**: Derived ratings from credit scores using industry standard bins
3. **PD Estimation**: Estimated PD from delinquency status and loan characteristics
4. **Configuration**: Realistic scenarios based on typical mortgage portfolio characteristics
5. **Analysis**: Monte Carlo simulation provides full distribution of RWA outcomes
6. **Interpretation**: Focus on mean for planning, P95 for stress, std for uncertainty

### Best Practices for Fannie Mae Data

✅ **Filter to recent data** (last 12-24 months) for representative portfolio
✅ **Handle missing values** appropriately (impute or exclude)
✅ **Validate rating distributions** match your internal policy
✅ **Calibrate PD estimates** to actual default rates in your portfolio
✅ **Document assumptions** clearly for audit trail
✅ **Run sensitivity analysis** on key parameters (AUC, LGD, default rate)

### Further Reading

- **Fannie Mae Documentation**: [Single-Family Loan Data](https://capitalmarkets.fanniemae.com)
- **Basel Framework**: [BCBS239 - AIRB Approach](https://www.bis.org/bcbs/publ/d347.pdf)
- **IRBStudio Docs**: [User Guide](user_guide.md) | [API Reference](api_reference.md)

### Getting Help

- **GitHub Issues**: [Report issues or ask questions](https://github.com/jacekkrawiec/IRBStudio/issues)
- **Example Notebooks**: See `notebooks/freddie_mac_sample_dataset.ipynb` for similar workflow

---

*This tutorial demonstrates IRBStudio capabilities with realistic data. Always validate results with your institution's model validation team before making business decisions.*

*Last Updated: January 2025*
