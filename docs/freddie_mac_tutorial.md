# IRBStudio Tutorial: Analyzing Freddie Mac Mortgage Data

This tutorial demonstrates how to use IRBStudio with real-world mortgage portfolio data, specifically using **Freddie Mac's Single-Family Loan-Level Dataset**.

Freddie Mac has been the primary data source used throughout IRBStudio's development and testing, making this tutorial closely aligned with the project's examples and test cases.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Freddie Mac Data](#understanding-freddie-mac-data)
3. [Quick Start: Using Pre-Prepared Data](#quick-start-using-pre-prepared-data)
4. [Data Preparation from Raw Files](#data-preparation-from-raw-files)
5. [Configuration Setup](#configuration-setup)
6. [Running the Analysis](#running-the-analysis)
7. [Interpreting Results](#interpreting-results)
8. [Advanced Examples](#advanced-examples)
9. [Complete Example Script](#complete-example-script)

---

## Introduction

### About Freddie Mac Data

Freddie Mac publishes anonymized Single-Family Loan-Level Dataset that includes:
- **Origination Data**: Loan characteristics at origination (credit score, LTV, DTI, property type, etc.)
- **Performance Data**: Monthly loan performance updates (balance, delinquency, modifications, dispositions)

### Dataset Source

**Freddie Mac Single-Family Loan-Level Dataset**
- **Source**: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
- **Format**: Pipe-delimited text files (|) without headers
- **Size**: Millions of loans, decades of performance history
- **Cost**: Free (registration required)
- **Update Frequency**: Quarterly

### Tutorial Scope

This tutorial covers:
1. ✅ Understanding Freddie Mac data structure
2. ✅ Transforming raw data to IRBStudio format
3. ✅ Running AIRB scenario analysis
4. ✅ Interpreting capital impact results
5. ✅ Advanced analytics (vintages, geography, segments)

---

## Understanding Freddie Mac Data

### File Structure

Freddie Mac provides two file types per quarter:

#### 1. Origination Files (`sample_orig_YYYY.txt`)

Contains **27 pipe-delimited columns** with loan characteristics at origination.

**Key Columns:**

| Position | Field Name | Description | Example Value |
|----------|------------|-------------|---------------|
| 1 | Credit Score | FICO score (300-850) | 812 |
| 2 | First Payment Date | YYYYMM format | 202403 |
| 8 | Occupancy Status | P=Primary, S=Second Home, I=Investment | P |
| 11 | Original UPB | Original loan amount in dollars | 246000 |
| 12 | Original LTV | Loan-to-Value ratio (percentage) | 38 |
| 16 | Product Type | FRM=Fixed Rate Mortgage, ARM=Adjustable | FRM |
| 17 | Property State | US state (2-letter code) | IL |
| 18 | Property Type | SF=Single Family, CO=Condo, etc. | SF |
| **20** | **Loan Sequence Number** | **Unique identifier** | **F24Q10000019** |
| 21 | Loan Purpose | P=Purchase, C=Cash-out Refi, N=No cash-out Refi | N |
| 22 | Original Loan Term | Term in months | 360 |

**Sample Line:**
```
812|202403|N|205402|16984|000|1|P|38|23|246000|38|7.375|R|N|FRM|IL|SF|60600|F24Q10000019|N|360|02|Other sellers|Other servicers|||9||2|N|7
```

#### 2. Performance Files (`sample_svcg_YYYY.txt`)

Contains **31 pipe-delimited columns** with monthly performance updates.

**Key Columns:**

| Position | Field Name | Description | Example Value |
|----------|------------|-------------|---------------|
| **1** | **Loan Sequence Number** | **Unique identifier (matches origination)** | **F24Q10000019** |
| **2** | **Monthly Reporting Period** | **YYYYMM format** | **202402** |
| **3** | **Current Actual UPB** | **Current outstanding balance** | **246000.00** |
| **4** | **Current Loan Delinquency Status** | **0=Current, 1=30DPD, 2=60DPD, 3+=90+DPD** | **0** |
| 5 | Loan Age | Months since origination | 000 |
| 6 | Remaining Months to Maturity | Months until loan maturity | 360 |
| 9 | Zero Balance Code | 01=Prepaid, 03=Foreclosure, 09=REO | (empty) |
| 11 | Current Interest Rate | Current rate (percentage) | 7.375 |

**Sample Line:**
```
F24Q10000019|202402|246000.00|0|000|360|||||7.375|0.00||||||||||||||40||||||246000.00
```

### IRBStudio Field Mapping

To use Freddie Mac data with IRBStudio, we need to map fields:

| IRBStudio Field | Freddie Mac Source | File | Transformation |
|-----------------|-------------------|------|----------------|
| `loan_id` | Loan Sequence Number | Both | Direct copy |
| `balance` | Current Actual UPB | Performance | Convert to numeric |
| `reporting_date` | Monthly Reporting Period | Performance | Parse YYYYMM → datetime |
| `default_flag` | Current Loan Delinquency Status | Performance | `≥ 3` = 1 (default), else 0 |
| `into_default_flag` | Current Loan Delinquency Status | Performance | Transition from 0-2 → 3+ |
| `score` | Credit Score | Origination | Normalize: `1 - (FICO-300)/550` |
| `rating` | Credit Score | Origination | Bin FICO into rating grades |
| `pd` | **Derived** | Multiple | Estimate from rating + LTV + delinquency |
| `ltv` | Original LTV | Origination | Direct copy |

### Key Challenge: Estimating PD

Freddie Mac data **does NOT include**:
- ❌ PD (Probability of Default) values
- ❌ Internal rating grades
- ❌ Normalized scores (only raw FICO 300-850)

**Solution**: We derive these fields using the following approach:

```python
# 1. Normalize FICO score to 0-1 scale (0=best, 1=worst)
normalized_score = 1 - ((fico_score - 300) / 550)
normalized_score = np.clip(normalized_score, 0, 1)

# 2. Bin FICO into rating grades
if fico >= 780: rating = 'AAA'
elif fico >= 740: rating = 'AA'
elif fico >= 700: rating = 'A'
elif fico >= 660: rating = 'BBB'
elif fico >= 620: rating = 'BB'
elif fico >= 580: rating = 'B'
else: rating = 'CCC'

# 3. Estimate PD from rating + risk adjustments
base_pd = rating_pd_map[rating]  # e.g., 'A' = 0.005 (0.5%)
ltv_adjustment = 1 + 0.3 * (ltv - 80) / 20  # Higher LTV = higher risk
delinq_multiplier = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0}[delinq_status]
pd = base_pd * ltv_adjustment * delinq_multiplier
pd = np.clip(pd, 0.0001, 0.50)  # Keep within reasonable bounds
```

---

## Quick Start: Using Pre-Prepared Data

IRBStudio includes pre-prepared Freddie Mac sample data. This is the **fastest way** to get started:

### Option 1: Run Example Script

```bash
cd examples
python freddie_mac_dashboard_example.py
```

This script:
1. ✅ Loads pre-prepared `data/sample_portfolio_data_fm.csv`
2. ✅ Runs AIRB scenario comparison (Current vs Enhanced vs Stress)
3. ✅ Generates interactive HTML dashboard
4. ✅ Outputs capital impact analysis

### Option 2: Use High-Level API

```python
from irbstudio import run_scenario_comparison

# Use the included sample dataset
results = run_scenario_comparison(
    config_path="examples/sample_config.yaml",
    portfolio_path="data/sample_portfolio_data_fm.csv",
    n_iterations=1000,
    random_seed=42,
    output_dir="results/freddie_mac_quick_start"
)

# View results
print("\n" + "="*60)
print("FREDDIE MAC PORTFOLIO ANALYSIS RESULTS")
print("="*60)

for scenario_name, scenario_results in results.items():
    if scenario_name != 'capital_delta':
        airb_mean = scenario_results['AIRB']['mean']
        airb_p95 = scenario_results['AIRB']['percentiles']['P95']
        print(f"\n{scenario_name}:")
        print(f"  Mean RWA: ${airb_mean:,.0f}")
        print(f"  P95 RWA:  ${airb_p95:,.0f}")

if 'capital_delta' in results:
    print(f"\nCapital Savings: ${results['capital_delta']:,.0f}")

print("\n" + "="*60)
print("Dashboard: results/freddie_mac_quick_start/scenario_comparison_dashboard.html")
print("="*60)
```

---

## Data Preparation from Raw Files

If you want to prepare data from **raw Freddie Mac files**, follow these steps:

### Step 1: Download Freddie Mac Data

1. **Register** at https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
2. **Download** quarterly files (e.g., 2024 Q1, Q2, Q3, Q4)
3. **Extract** to your `data/FM/` directory:
   - `sample_orig_2024.txt` (origination data)
   - `sample_svcg_2024.txt` (performance data)

### Step 2: Load and Merge Data

```python
import pandas as pd
import numpy as np

# Define column names (Freddie Mac files have NO headers)
orig_columns = [
    'credit_score', 'first_payment_date', 'first_time_homebuyer_flag',
    'maturity_date', 'msa', 'mi_percentage', 'number_of_units',
    'occupancy_status', 'original_cltv', 'original_dti', 'original_upb',
    'original_ltv', 'original_interest_rate', 'channel',
    'prepayment_penalty_flag', 'product_type', 'property_state',
    'property_type', 'postal_code', 'loan_sequence_number',
    'loan_purpose', 'original_loan_term', 'number_of_borrowers',
    'seller_name', 'servicer_name', 'super_conforming_flag',
    'pre_harp_loan_sequence_number'
]

perf_columns = [
    'loan_sequence_number', 'monthly_reporting_period',
    'current_actual_upb', 'current_loan_delinquency_status',
    'loan_age', 'remaining_months_to_maturity',
    'repurchase_flag', 'modification_flag', 'zero_balance_code',
    'zero_balance_effective_date', 'current_interest_rate',
    'current_deferred_upb', 'due_date_of_last_paid_installment',
    'mi_recoveries', 'net_sales_proceeds', 'non_mi_recoveries',
    'expenses', 'legal_costs', 'maintenance_costs',
    'taxes_and_insurance', 'miscellaneous_expenses',
    'actual_loss_calculation', 'modification_cost',
    'step_modification_flag', 'deferred_payment_plan',
    'estimated_ltv', 'zero_balance_removal_upb',
    'delinquent_accrued_interest', 'delinquency_due_to_disaster',
    'borrower_assistance_status_code', 'current_month_modification_cost'
]

# Load data with pipe delimiter
print("Loading Freddie Mac data...")
orig_df = pd.read_csv(
    'data/FM/sample_orig_2024.txt',
    sep='|',
    names=orig_columns,
    header=None,
    low_memory=False
)

perf_df = pd.read_csv(
    'data/FM/sample_svcg_2024.txt',
    sep='|',
    names=perf_columns,
    header=None,
    low_memory=False
)

print(f"Origination records: {len(orig_df):,}")
print(f"Performance records: {len(perf_df):,}")

# Merge on loan_sequence_number (inner join keeps only matching loans)
portfolio = perf_df.merge(
    orig_df,
    on='loan_sequence_number',
    how='inner'
)

print(f"Merged records: {len(portfolio):,}")
```

### Step 3: Transform to IRBStudio Format

```python
# ===================================================================
# STEP 3A: Parse Dates
# ===================================================================
# Freddie Mac uses YYYYMM format (e.g., 202403 = March 2024)
portfolio['reporting_date'] = pd.to_datetime(
    portfolio['monthly_reporting_period'].astype(str),
    format='%Y%m',
    errors='coerce'
)

# ===================================================================
# STEP 3B: Normalize Credit Score
# ===================================================================
# Convert to numeric and normalize to 0-1 scale (0=best, 1=worst)
portfolio['credit_score'] = pd.to_numeric(
    portfolio['credit_score'],
    errors='coerce'
)

portfolio['normalized_score'] = 1 - (
    (portfolio['credit_score'] - 300) / 550
)
portfolio['normalized_score'] = portfolio['normalized_score'].clip(0, 1)

# ===================================================================
# STEP 3C: Assign Rating Grades
# ===================================================================
def assign_rating(fico):
    """Assign rating grade based on FICO score"""
    if pd.isna(fico):
        return 'B'  # Default to middle rating for missing scores
    if fico >= 780: return 'AAA'
    elif fico >= 740: return 'AA'
    elif fico >= 700: return 'A'
    elif fico >= 660: return 'BBB'
    elif fico >= 620: return 'BB'
    elif fico >= 580: return 'B'
    else: return 'CCC'

portfolio['rating'] = portfolio['credit_score'].apply(assign_rating)

# ===================================================================
# STEP 3D: Estimate PD
# ===================================================================
# Base PD by rating grade
rating_pd = {
    'AAA': 0.0005,  # 0.05%
    'AA': 0.001,    # 0.1%
    'A': 0.005,     # 0.5%
    'BBB': 0.01,    # 1%
    'BB': 0.03,     # 3%
    'B': 0.05,      # 5%
    'CCC': 0.10     # 10%
}
portfolio['pd'] = portfolio['rating'].map(rating_pd)

# Adjust for LTV (higher LTV = higher risk)
portfolio['original_ltv'] = pd.to_numeric(
    portfolio['original_ltv'],
    errors='coerce'
)
ltv_adj = 1 + 0.3 * (portfolio['original_ltv'] - 80) / 20
ltv_adj = ltv_adj.fillna(1.0)
portfolio['pd'] = portfolio['pd'] * ltv_adj

# Adjust for current delinquency status (higher delinquency = higher risk)
portfolio['delinq'] = portfolio['current_loan_delinquency_status'].fillna('0').astype(str).str.strip()
delinq_multiplier = {
    '0': 1.0,   # Current
    '1': 2.0,   # 30 days past due
    '2': 4.0,   # 60 days past due
    '3': 8.0,   # 90+ days past due
}
portfolio['delinq_mult'] = portfolio['delinq'].apply(
    lambda x: delinq_multiplier.get(x, 1.0)
)
portfolio['pd'] = portfolio['pd'] * portfolio['delinq_mult']

# Clip PD to reasonable bounds
portfolio['pd'] = portfolio['pd'].clip(0.0001, 0.50)

# ===================================================================
# STEP 3E: Create Default Flags
# ===================================================================
# default_flag: 1 if loan is currently in default (90+ DPD), else 0
portfolio['default_flag'] = portfolio['delinq'].apply(
    lambda x: 1 if x not in ['0', '1', '2', ''] else 0
)

# into_default_flag: 1 if loan newly defaulted this period
portfolio = portfolio.sort_values(['loan_sequence_number', 'reporting_date'])
portfolio['prev_default'] = portfolio.groupby('loan_sequence_number')['default_flag'].shift(1).fillna(0)
portfolio['into_default_flag'] = (
    (portfolio['default_flag'] == 1) & (portfolio['prev_default'] == 0)
).astype(int)

print("\nTransformed Portfolio Sample:")
print(portfolio[[
    'loan_sequence_number', 'reporting_date', 'current_actual_upb',
    'credit_score', 'rating', 'pd', 'normalized_score', 'default_flag'
]].head(10))
```

### Step 4: Prepare Final Dataset

```python
# ===================================================================
# STEP 4: Select and Rename Columns to IRBStudio Format
# ===================================================================
irbstudio_portfolio = portfolio[[
    'loan_sequence_number',
    'current_actual_upb',
    'pd',
    'normalized_score',
    'rating',
    'reporting_date',
    'default_flag',
    'into_default_flag',
    'original_ltv'
]].copy()

# Rename to IRBStudio canonical names
irbstudio_portfolio.columns = [
    'loan_id',
    'balance',
    'pd',
    'score',
    'rating',
    'reporting_date',
    'default_flag',
    'into_default_flag',
    'ltv'
]

# ===================================================================
# STEP 5: Clean and Filter Data
# ===================================================================
# Convert balance to numeric
irbstudio_portfolio['balance'] = pd.to_numeric(
    irbstudio_portfolio['balance'],
    errors='coerce'
)

# Remove zero balance loans
irbstudio_portfolio = irbstudio_portfolio[
    irbstudio_portfolio['balance'] > 0
]

# Remove loans with missing critical data
irbstudio_portfolio = irbstudio_portfolio.dropna(
    subset=['balance', 'pd', 'score', 'rating', 'reporting_date']
)

# Filter to recent 12 months
if not irbstudio_portfolio['reporting_date'].isna().all():
    max_date = irbstudio_portfolio['reporting_date'].max()
    cutoff_date = max_date - pd.DateOffset(months=12)
    irbstudio_portfolio = irbstudio_portfolio[
        irbstudio_portfolio['reporting_date'] >= cutoff_date
    ]

# ===================================================================
# STEP 6: Save Prepared Data
# ===================================================================
output_path = 'data/freddie_mac_prepared.csv'
irbstudio_portfolio.to_csv(output_path, index=False)

print(f"\n{'='*70}")
print("FREDDIE MAC DATA PREPARATION COMPLETE")
print(f"{'='*70}")
print(f"Output file:     {output_path}")
print(f"Total records:   {len(irbstudio_portfolio):,}")
print(f"Unique loans:    {irbstudio_portfolio['loan_id'].nunique():,}")
print(f"Date range:      {irbstudio_portfolio['reporting_date'].min()} to {irbstudio_portfolio['reporting_date'].max()}")
print(f"Total exposure:  ${irbstudio_portfolio['balance'].sum():,.0f}")
print(f"\nRating Distribution:")
print(irbstudio_portfolio['rating'].value_counts().sort_index())
print(f"{'='*70}")
```

---

## Configuration Setup

Create a file named `freddie_mac_config.yaml`:

```yaml
# ===================================================================
# FREDDIE MAC PORTFOLIO CONFIGURATION
# ===================================================================

# Column mapping: Map your data columns to IRBStudio canonical fields
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

# Regulatory parameters for US residential mortgages
regulatory:
  jurisdiction: us
  asset_correlation: 0.15      # Basel standard for residential mortgages
  confidence_level: 0.999      # 99.9% confidence level for capital

# ===================================================================
# SCENARIOS
# ===================================================================

scenarios:
  # Scenario 1: Current State
  - name: "Current Model"
    description: "Baseline FICO-based PD model (AUC ~0.72)"
    pd_auc: 0.72                    # Typical AUC for FICO-only models
    portfolio_default_rate: 0.015   # 1.5% (typical for prime mortgages)
    lgd: 0.25                       # 25% Loss Given Default
    new_loan_rate: 0.08             # 8% of portfolio is new originations
    rating_pd_map:
      AAA: 0.0005
      AA: 0.001
      A: 0.005
      BBB: 0.01
      BB: 0.03
      B: 0.05
      CCC: 0.10

  # Scenario 2: Model Enhancement
  - name: "Enhanced Model"
    description: "Improved model with additional data sources (AUC ~0.78)"
    pd_auc: 0.78                    # Better discrimination with payment history, DTI, etc.
    portfolio_default_rate: 0.015   # Same default rate, better ranking
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

  # Scenario 3: Economic Stress
  - name: "Stress Scenario"
    description: "Recession scenario with elevated defaults"
    pd_auc: 0.68                    # Model discrimination degrades under stress
    portfolio_default_rate: 0.04    # 4% default rate (recession level)
    lgd: 0.35                       # Higher LGD (depressed housing, longer foreclosures)
    new_loan_rate: 0.03             # Lower origination activity
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

### Example 1: Full Scenario Comparison

```python
from irbstudio import run_scenario_comparison

# Run comparison of all three scenarios
results = run_scenario_comparison(
    config_path="freddie_mac_config.yaml",
    portfolio_path="data/freddie_mac_prepared.csv",
    n_iterations=5000,  # Higher iterations for production analysis
    random_seed=42,
    output_dir="results/freddie_mac_analysis"
)

# Print comprehensive results
print("\n" + "="*70)
print("FREDDIE MAC PORTFOLIO: AIRB CAPITAL IMPACT ANALYSIS")
print("="*70)

for scenario_name in ['Current Model', 'Enhanced Model', 'Stress Scenario']:
    if scenario_name in results:
        airb_stats = results[scenario_name]['AIRB']
        
        print(f"\n{scenario_name}:")
        print(f"  Mean RWA:       ${airb_stats['mean']:>15,.0f}")
        print(f"  Std Dev:        ${airb_stats['std']:>15,.0f}")
        print(f"  Median RWA:     ${airb_stats['median']:>15,.0f}")
        print(f"  P95 RWA:        ${airb_stats['percentiles']['P95']:>15,.0f}")
        print(f"  P99 RWA:        ${airb_stats['percentiles']['P99']:>15,.0f}")

# Calculate and display capital impact
if 'capital_delta' in results:
    savings = results['capital_delta']
    current_rwa = results['Current Model']['AIRB']['mean']
    enhanced_rwa = results['Enhanced Model']['AIRB']['mean']
    
    print(f"\n{'-'*70}")
    print("CAPITAL IMPACT OF MODEL ENHANCEMENT:")
    print(f"  Current Model RWA:      ${current_rwa:>15,.0f}")
    print(f"  Enhanced Model RWA:     ${enhanced_rwa:>15,.0f}")
    print(f"  RWA Reduction:          ${current_rwa - enhanced_rwa:>15,.0f}")
    print(f"  Capital Savings (8%):   ${savings:>15,.0f}")
    print(f"  Percentage Reduction:   {((current_rwa - enhanced_rwa) / current_rwa * 100):>14.1f}%")

print("\n" + "="*70)
print(f"Interactive Dashboard Saved:")
print(f"  → results/freddie_mac_analysis/scenario_comparison_dashboard.html")
print("="*70)
```

### Example 2: AIRB vs Standardized Approach

```python
from irbstudio import load_config
from irbstudio.data.loader import load_portfolio
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator
from irbstudio.engine.integrated_analysis import IntegratedAnalysis
from irbstudio.engine.mortgage import AIRBMortgageCalculator, SAMortgageCalculator

# Load configuration and data
config = load_config("freddie_mac_config.yaml")
portfolio_df = load_portfolio(
    "data/freddie_mac_prepared.csv",
    config.column_mapping
)

# Create analysis engine
analysis = IntegratedAnalysis()

# Add both AIRB and SA calculators
analysis.add_calculator('AIRB', AIRBMortgageCalculator(
    regulatory_params={
        'lgd': 0.25,
        'asset_correlation': 0.15,
        'confidence_level': 0.999
    }
))
analysis.add_calculator('SA', SAMortgageCalculator())

# Create simulator for current model scenario
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

# Add scenario and run
analysis.add_scenario('Current Model', simulator, n_iterations=2000)
results = analysis.run_scenario(
    scenario_name='Current Model',
    random_seed=42,
    application_start_date='2024-01-01'
)

# Compare AIRB vs SA
airb_mean = results['AIRB']['mean']
sa_mean = results['SA']['mean']
rwa_reduction = sa_mean - airb_mean
capital_savings = rwa_reduction * 0.08

print("\n" + "="*70)
print("AIRB vs STANDARDIZED APPROACH COMPARISON")
print("="*70)
print(f"\nStandardized Approach (SA):")
print(f"  Mean RWA:               ${sa_mean:>15,.0f}")
print(f"\nAdvanced IRB (AIRB):")
print(f"  Mean RWA:               ${airb_mean:>15,.0f}")
print(f"\nBenefit of AIRB Adoption:")
print(f"  RWA Reduction:          ${rwa_reduction:>15,.0f}")
print(f"  Capital Freed (8%):     ${capital_savings:>15,.0f}")
print(f"  Percentage Reduction:   {(rwa_reduction / sa_mean * 100):>14.1f}%")
print("="*70)
```

---

## Interpreting Results

### Key Metrics Explained

#### 1. Mean RWA
- **Definition**: Average RWA across all Monte Carlo simulations
- **Use**: Primary metric for capital planning and budgeting
- **Typical Values** (for $1B mortgage portfolio):
  - **Conservative AIRB**: $300M - $450M (30-45% RWA density)
  - **Moderate AIRB**: $200M - $300M (20-30%)
  - **Optimized AIRB**: $150M - $200M (15-20%)
  - **SA Baseline**: $350M - $500M (35-50%)

#### 2. Standard Deviation
- **Definition**: Measure of RWA volatility/uncertainty
- **Use**: Understanding model risk and capital buffer sizing
- **Interpretation**:
  - **Low (< 5% of mean)**: Stable portfolio, low model uncertainty
  - **Medium (5-15%)**: Typical for well-established mortgage portfolios
  - **High (> 15%)**: High uncertainty - consider additional validation

#### 3. Percentiles (P95, P99)
- **P95**: 95th percentile - only 5% of simulations exceed this value
- **P99**: 99th percentile - only 1% of simulations exceed this value
- **Use Cases**:
  - **P95**: Stress testing, ICAAP capital adequacy scenarios
  - **P99**: Severe stress, regulatory capital buffer determination

### Dashboard Visualization Guide

The generated HTML dashboard (`scenario_comparison_dashboard.html`) includes:

**1. RWA Distribution Plots**
- **Histogram**: Shows frequency of RWA outcomes across simulations
- **KDE Overlay**: Smooth probability density curve
- **Percentile Markers**: Vertical lines at P5, P50 (median), P95
- **Interpretation**:
  - **Symmetric distribution** = Stable, well-behaved model
  - **Right skew** = Tail risk (some very high RWA outcomes)
  - **Bimodal** = Portfolio segments behaving differently

**2. Scenario Comparison Charts**
- **Grouped Bars**: Mean RWA for each scenario side-by-side
- **Error Bars**: ±1 standard deviation range
- **Use**: Quick visual assessment of scenario differences

**3. Statistical Summary Tables**

Example output for Freddie Mac portfolio:

| Metric | Current Model | Enhanced Model | Stress Scenario |
|--------|---------------|----------------|-----------------|
| Mean RWA | $287M | $251M | $462M |
| Std Dev | $16M | $14M | $39M |
| Median | $285M | $249M | $458M |
| P95 RWA | $315M | $275M | $531M |
| Skewness | 0.14 | 0.11 | 0.32 |
| Capital (8%) | $23.0M | $20.1M | $37.0M |

### Business Case Calculation

#### Model Improvement ROI

```
Portfolio:           $1.0B
Current Model:       $287M RWA → $23.0M capital (8% × RWA)
Enhanced Model:      $251M RWA → $20.1M capital

RWA Reduction:       $36M
Capital Freed:       $2.9M

Model Development:
  Initial Cost:      $500K (one-time)
  Annual Maintenance: $100K/year

ROI Analysis:
  Annual Benefit:    $2.9M (capital freed)
  Annual Cost:       $0.1M (maintenance)
  Net Annual:        $2.8M
  
  Payback Period:    $0.5M / $2.8M = 0.18 years (~2 months!)
  5-Year NPV (10%):  $9.55M
  
Conclusion: STRONG business case for model enhancement
```

#### AIRB vs SA Decision Framework

```
Portfolio:           $1.0B residential mortgages
Current Approach:    Standardized (SA)

Comparison:
  SA Mean RWA:       $450M → $36.0M capital
  AIRB Mean RWA:     $287M → $23.0M capital
  
  Capital Freed:     $13.0M

AIRB Implementation:
  Initial Investment: $2-5M (systems, models, validation, training)
  Annual Costs:       $0.5-1M (operations, validation, reporting)
  
Break-Even:          2-5M / 13M = 0.15-0.38 years (2-5 months)

ROI (5 years):       ($13M × 5 - $5M - $1M × 5) / ($5M + $1M × 5) 
                     = $50M / $10M = 500%

Conclusion: AIRB adoption HIGHLY beneficial for this portfolio size
```

---

## Advanced Examples

### Example 3: Vintage Analysis

Analyze different loan origination years separately:

```python
# Assuming we can derive origination_year from first_payment_date
portfolio_df['first_payment_date'] = pd.to_datetime(
    orig_df['first_payment_date'].astype(str),
    format='%Y%m'
)
portfolio_df['origination_year'] = portfolio_df['first_payment_date'].dt.year

vintages_rwa = {}

for year in [2021, 2022, 2023, 2024]:
    vintage_df = portfolio_df[portfolio_df['origination_year'] == year]
    
    if len(vintage_df) < 100:  # Skip small samples
        print(f"Skipping {year}: insufficient data ({len(vintage_df)} loans)")
        continue
    
    simulator = PortfolioSimulator(
        portfolio_df=vintage_df,
        score_to_rating_bounds={'A': (0.03, 0.10), 'B': (0.10, 0.20)},
        target_auc=0.75
    )
    
    analysis.add_scenario(f'Vintage_{year}', simulator, n_iterations=1000)
    results = analysis.run_scenario(f'Vintage_{year}', random_seed=42)
    
    vintages_rwa[year] = {
        'mean_rwa': results['AIRB']['mean'],
        'exposure': vintage_df['balance'].sum(),
        'rwa_density': results['AIRB']['mean'] / vintage_df['balance'].sum(),
        'avg_fico': vintage_df['credit_score'].mean(),
        'avg_ltv': vintage_df['ltv'].mean()
    }

# Print vintage comparison
print("\n" + "="*70)
print("RWA BY ORIGINATION VINTAGE")
print("="*70)
print(f"{'Year':<6} {'Exposure':>12} {'Mean RWA':>12} {'Density':>9} {'Avg FICO':>10} {'Avg LTV':>9}")
print("-" * 70)
for year, metrics in sorted(vintages_rwa.items()):
    print(f"{year:<6} ${metrics['exposure']:>11,.0f} ${metrics['mean_rwa']:>11,.0f} "
          f"{metrics['rwa_density']:>8.1%} {metrics['avg_fico']:>10.0f} {metrics['avg_ltv']:>8.0f}%")
print("="*70)
```

### Example 4: Geographic Segmentation

Analyze RWA by state or region:

```python
# Assuming property_state is available in portfolio_df
states = portfolio_df['property_state'].value_counts().head(10).index

state_rwa = {}

for state in states:
    state_df = portfolio_df[portfolio_df['property_state'] == state]
    
    if len(state_df) < 100:  # Skip small segments
        continue
    
    simulator = PortfolioSimulator(
        portfolio_df=state_df,
        target_auc=0.75
    )
    
    analysis.add_scenario(f'State_{state}', simulator, n_iterations=500)
    results = analysis.run_scenario(f'State_{state}', random_seed=42)
    
    state_rwa[state] = {
        'mean_rwa': results['AIRB']['mean'],
        'exposure': state_df['balance'].sum(),
        'rwa_density': results['AIRB']['mean'] / state_df['balance'].sum(),
        'avg_fico': state_df['credit_score'].mean(),
        'avg_ltv': state_df['ltv'].mean(),
        'default_rate': state_df['default_flag'].mean()
    }

# Print geographic analysis
print("\n" + "="*75)
print("RWA BY STATE (TOP 10 BY EXPOSURE)")
print("="*75)
print(f"{'State':<6} {'Exposure':>12} {'RWA':>12} {'Density':>9} {'FICO':>7} {'LTV':>6} {'Def%':>6}")
print("-" * 75)
for state, metrics in sorted(state_rwa.items(), key=lambda x: x[1]['exposure'], reverse=True):
    print(f"{state:<6} ${metrics['exposure']:>11,.0f} ${metrics['mean_rwa']:>11,.0f} "
          f"{metrics['rwa_density']:>8.1%} {metrics['avg_fico']:>7.0f} {metrics['avg_ltv']:>5.0f}% "
          f"{metrics['default_rate']:>5.1%}")
print("="*75)
```

### Example 5: Product Type Comparison

Compare Fixed Rate Mortgages (FRM) vs Adjustable Rate Mortgages (ARM):

```python
# Assuming product_type is available
products = ['FRM', 'ARM']

for product in products:
    product_df = portfolio_df[portfolio_df['product_type'] == product]
    
    if len(product_df) < 50:
        print(f"Skipping {product}: insufficient data")
        continue
    
    simulator = PortfolioSimulator(
        portfolio_df=product_df,
        target_auc=0.75
    )
    
    analysis.add_scenario(f'Product_{product}', simulator, n_iterations=1000)
    results = analysis.run_scenario(f'Product_{product}', random_seed=42)
    
    print(f"\n{product} Product Analysis:")
    print(f"  Loan Count:      {len(product_df):,}")
    print(f"  Total Exposure:  ${product_df['balance'].sum():,.0f}")
    print(f"  Mean RWA:        ${results['AIRB']['mean']:,.0f}")
    print(f"  RWA Density:     {results['AIRB']['mean'] / product_df['balance'].sum():.1%}")
    print(f"  Avg Interest:    {product_df['current_interest_rate'].mean():.2f}%")
```

---

## Complete Example Script

Here's a production-ready script that combines all steps:

```python
#!/usr/bin/env python3
"""
Complete Freddie Mac Data Preparation and Analysis Script
Transforms raw Freddie Mac data and runs AIRB scenario analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from irbstudio import run_scenario_comparison


def prepare_freddie_mac_data(orig_file, perf_file, output_file, months_back=12):
    """Prepare Freddie Mac data for IRBStudio"""
    
    # Column definitions
    orig_columns = [
        'credit_score', 'first_payment_date', 'first_time_homebuyer_flag',
        'maturity_date', 'msa', 'mi_percentage', 'number_of_units',
        'occupancy_status', 'original_cltv', 'original_dti', 'original_upb',
        'original_ltv', 'original_interest_rate', 'channel',
        'prepayment_penalty_flag', 'product_type', 'property_state',
        'property_type', 'postal_code', 'loan_sequence_number',
        'loan_purpose', 'original_loan_term', 'number_of_borrowers',
        'seller_name', 'servicer_name', 'super_conforming_flag',
        'pre_harp_loan_sequence_number'
    ]
    
    perf_columns = [
        'loan_sequence_number', 'monthly_reporting_period',
        'current_actual_upb', 'current_loan_delinquency_status',
        'loan_age', 'remaining_months_to_maturity',
        'repurchase_flag', 'modification_flag', 'zero_balance_code',
        'zero_balance_effective_date', 'current_interest_rate',
        'current_deferred_upb', 'due_date_of_last_paid_installment',
        'mi_recoveries', 'net_sales_proceeds', 'non_mi_recoveries',
        'expenses', 'legal_costs', 'maintenance_costs',
        'taxes_and_insurance', 'miscellaneous_expenses',
        'actual_loss_calculation', 'modification_cost',
        'step_modification_flag', 'deferred_payment_plan',
        'estimated_ltv', 'zero_balance_removal_upb',
        'delinquent_accrued_interest', 'delinquency_due_to_disaster',
        'borrower_assistance_status_code', 'current_month_modification_cost'
    ]
    
    # Load data
    print("Loading Freddie Mac data...")
    orig_df = pd.read_csv(orig_file, sep='|', names=orig_columns, header=None, low_memory=False)
    perf_df = pd.read_csv(perf_file, sep='|', names=perf_columns, header=None, low_memory=False)
    
    # Merge
    portfolio = perf_df.merge(orig_df, on='loan_sequence_number', how='inner')
    
    # Transform dates
    portfolio['reporting_date'] = pd.to_datetime(
        portfolio['monthly_reporting_period'].astype(str),
        format='%Y%m',
        errors='coerce'
    )
    
    # Filter to recent data
    max_date = portfolio['reporting_date'].max()
    cutoff_date = max_date - pd.DateOffset(months=months_back)
    portfolio = portfolio[portfolio['reporting_date'] >= cutoff_date]
    
    # Normalize score
    portfolio['credit_score'] = pd.to_numeric(portfolio['credit_score'], errors='coerce')
    portfolio['normalized_score'] = 1 - ((portfolio['credit_score'] - 300) / 550)
    portfolio['normalized_score'] = portfolio['normalized_score'].clip(0, 1)
    
    # Assign ratings
    def assign_rating(score):
        if pd.isna(score): return 'B'
        if score >= 780: return 'AAA'
        elif score >= 740: return 'AA'
        elif score >= 700: return 'A'
        elif score >= 660: return 'BBB'
        elif score >= 620: return 'BB'
        elif score >= 580: return 'B'
        else: return 'CCC'
    
    portfolio['rating'] = portfolio['credit_score'].apply(assign_rating)
    
    # Estimate PD
    rating_pd = {'AAA': 0.0005, 'AA': 0.001, 'A': 0.005, 'BBB': 0.01, 
                 'BB': 0.03, 'B': 0.05, 'CCC': 0.10}
    portfolio['pd'] = portfolio['rating'].map(rating_pd)
    
    # LTV adjustment
    portfolio['original_ltv'] = pd.to_numeric(portfolio['original_ltv'], errors='coerce')
    ltv_adj = 1 + 0.3 * (portfolio['original_ltv'] - 80) / 20
    portfolio['pd'] = portfolio['pd'] * ltv_adj.fillna(1.0)
    
    # Delinquency adjustment
    portfolio['delinq'] = portfolio['current_loan_delinquency_status'].fillna('0').astype(str).str.strip()
    delinq_mult = {'0': 1.0, '1': 2.0, '2': 4.0, '3': 8.0}
    portfolio['pd'] = portfolio['pd'] * portfolio['delinq'].apply(lambda x: delinq_mult.get(x, 1.0))
    portfolio['pd'] = portfolio['pd'].clip(0.0001, 0.50)
    
    # Default flags
    portfolio['default_flag'] = portfolio['delinq'].apply(lambda x: 1 if x not in ['0','1','2',''] else 0)
    portfolio = portfolio.sort_values(['loan_sequence_number', 'reporting_date'])
    portfolio['prev_default'] = portfolio.groupby('loan_sequence_number')['default_flag'].shift(1).fillna(0)
    portfolio['into_default_flag'] = ((portfolio['default_flag'] == 1) & (portfolio['prev_default'] == 0)).astype(int)
    
    # Prepare output
    output_df = portfolio[[
        'loan_sequence_number', 'current_actual_upb', 'pd', 'normalized_score',
        'rating', 'reporting_date', 'default_flag', 'into_default_flag', 'original_ltv'
    ]].copy()
    
    output_df.columns = ['loan_id', 'balance', 'pd', 'score', 'rating',
                         'reporting_date', 'default_flag', 'into_default_flag', 'ltv']
    
    # Clean
    output_df['balance'] = pd.to_numeric(output_df['balance'], errors='coerce')
    output_df = output_df[output_df['balance'] > 0]
    output_df = output_df.dropna(subset=['balance', 'pd', 'score', 'rating', 'reporting_date'])
    
    # Save
    output_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("PREPARATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output:      {output_file}")
    print(f"Records:     {len(output_df):,}")
    print(f"Unique loans: {output_df['loan_id'].nunique():,}")
    print(f"Date range:  {output_df['reporting_date'].min()} to {output_df['reporting_date'].max()}")
    print(f"Exposure:    ${output_df['balance'].sum():,.0f}")
    print(f"\nRating Distribution:")
    print(output_df['rating'].value_counts().sort_index())
    print(f"{'='*70}\n")
    
    return output_df


if __name__ == "__main__":
    # Step 1: Prepare data
    prepared_data = prepare_freddie_mac_data(
        orig_file='data/FM/sample_orig_2024.txt',
        perf_file='data/FM/sample_svcg_2024.txt',
        output_file='data/freddie_mac_prepared.csv',
        months_back=12
    )
    
    # Step 2: Run analysis
    print("Running AIRB scenario analysis...")
    results = run_scenario_comparison(
        config_path='freddie_mac_config.yaml',
        portfolio_path='data/freddie_mac_prepared.csv',
        n_iterations=5000,
        random_seed=42,
        output_dir='results/freddie_mac_analysis'
    )
    
    # Step 3: Display results
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Dashboard: results/freddie_mac_analysis/scenario_comparison_dashboard.html")
    print("="*70)
```

---

## Summary

### Key Takeaways

✅ **Freddie Mac is IRBStudio's primary test dataset**  
✅ **Pipe-delimited format (|) requires explicit column definitions**  
✅ **YYYYMM date format needs parsing to datetime**  
✅ **PD and ratings must be derived (not provided in raw data)**  
✅ **Rich loan characteristics enable advanced segmentation analysis**  
✅ **Multiple scenarios enable comprehensive capital impact assessment**

### Best Practices

1. ✅ **Filter to recent 12-24 months** for current portfolio representation
2. ✅ **Validate FICO distribution** (expect 700-750 average for prime mortgages)
3. ✅ **Exclude zero balance loans** from analysis
4. ✅ **Calibrate PD adjustments** to your institution's historical experience
5. ✅ **Document all assumptions** for audit trail and validation
6. ✅ **Run sensitivity analysis** on key parameters (AUC, LGD, default rate)
7. ✅ **Segment by vintage and geography** to identify risk concentrations

### Data Quality Checks

Before running production analysis:

```python
print("="*70)
print("DATA QUALITY REPORT")
print("="*70)
print(f"Missing credit scores:    {portfolio['credit_score'].isna().sum():,}")
print(f"Missing LTV:              {portfolio['original_ltv'].isna().sum():,}")
print(f"Zero balances:            {(portfolio['current_actual_upb'] == 0).sum():,}")
print(f"\nDelinquency Status Distribution:")
print(portfolio['delinq'].value_counts().sort_index())
print(f"\nCredit Score Statistics:")
print(portfolio['credit_score'].describe())
print(f"\nLTV Statistics:")
print(portfolio['original_ltv'].describe())
print("="*70)
```

### Resources

- **Freddie Mac Dataset**: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
- **IRBStudio Documentation**: [User Guide](user_guide.md) | [API Reference](api_reference.md)
- **Example Scripts**: `examples/freddie_mac_dashboard_example.py`
- **Sample Data**: `data/FM/sample_orig_2024.txt`, `data/FM/sample_svcg_2024.txt`
- **Basel Framework**: [BCBS 424 - AIRB Requirements](https://www.bis.org/bcbs/publ/d424.pdf)

### Getting Help

- **GitHub Issues**: [Report issues or ask questions](https://github.com/jacekkrawiec/IRBStudio/issues)
- **Example Notebooks**: See `notebooks/freddie_mac_sample_dataset.ipynb`
- **Email Support**: Contact the maintainer for technical assistance

---

*This tutorial demonstrates IRBStudio capabilities using Freddie Mac's loan-level dataset structure. Always validate results with your institution's model validation team before making business decisions.*

*Last Updated: October 2025*
