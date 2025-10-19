# IRBStudio Tutorial: Analyzing Freddie Mac Mortgage Data# IRBStudio Tutorial: Analyzing Freddie Mac Mortgage Data



This tutorial demonstrates how to use IRBStudio with real-world mortgage portfolio data, specifically using **Freddie Mac's Single-Family Loan-Level Dataset**.This tutorial demonstrates how to use IRBStudio with real-world mortgage portfolio data, specifically using the structure of Freddie Mac's Single-Family Loan-Level Dataset.



Freddie Mac has been the primary data source used throughout IRBStudio's development and testing, making this tutorial closely aligned with the project's examples and test cases.---



---## Table of Contents



## Table of Contents1. [Introduction](#introduction)

2. [Understanding Freddie Mac Data](#understanding-freddie-mac-data)

1. [Introduction](#introduction)3. [Data Preparation](#data-preparation)

2. [Understanding Freddie Mac Data](#understanding-freddie-mac-data)4. [Configuration Setup](#configuration-setup)

3. [Quick Start: Using Pre-Prepared Data](#quick-start-using-pre-prepared-data)5. [Running the Analysis](#running-the-analysis)

4. [Data Preparation from Raw Files](#data-preparation-from-raw-files)6. [Interpreting Results](#interpreting-results)

5. [Configuration Setup](#configuration-setup)7. [Advanced Examples](#advanced-examples)

6. [Running the Analysis](#running-the-analysis)

7. [Interpreting Results](#interpreting-results)---

8. [Advanced Examples](#advanced-examples)

## Introduction

---

### About Freddie Mac Data

## Introduction

Freddie Mac publishes anonymized Single-Family Loan-Level Dataset that includes:

### About Freddie Mac Data- **Origination Data**: Loan characteristics at origination

- **Performance Data**: Monthly loan performance updates

Freddie Mac publishes anonymized Single-Family Loan-Level Dataset that includes:

- **Origination Data**: Loan characteristics at origination (credit score, LTV, DTI, property type, etc.)This tutorial shows how to:

- **Performance Data**: Monthly loan performance updates (balance, delinquency, modifications, dispositions)1. Prepare Freddie Mac data for IRBStudio

2. Create appropriate configurations

### Dataset Source3. Run AIRB scenario analysis

4. Interpret capital impact results

**Freddie Mac Single-Family Loan-Level Dataset**

- **Source**: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset### Dataset Source

- **Format**: Pipe-delimited text files (|) without headers

- **Size**: Millions of loans, decades of performance history**Freddie Mac Single-Family Loan-Level Dataset**

- **Cost**: Free (registration required)- Source: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

- **Update Frequency**: Quarterly- Format: Pipe-delimited text files (origination + performance)

- Size: Millions of loans, decades of performance

### Tutorial Scope- Cost: Free (registration required)



This tutorial covers:---

1. ✅ Understanding Freddie Mac data structure

2. ✅ Transforming raw data to IRBStudio format## Understanding Freddie Mac Data

3. ✅ Running AIRB scenario analysis

4. ✅ Interpreting capital impact results### File Structure

5. ✅ Advanced analytics (vintages, geography, segments)

Fannie Mae provides two types of files:

---

**1. Origination Files** (`sample_orig_YYYY.txt`)

## Understanding Freddie Mac DataContains loan characteristics at origination:

- Credit score

### File Structure- LTV ratio

- DTI ratio

Freddie Mac provides two file types per quarter:- Loan purpose

- Property type

#### 1. Origination Files (`sample_orig_YYYY.txt`)- Number of borrowers

- First-time homebuyer flag

Contains 27 pipe-delimited columns with loan characteristics at origination:

**2. Servicing Files** (`sample_svcg_YYYY.txt`)

| Position | Field | Description | Example |Contains monthly performance updates:

|----------|-------|-------------|---------|- Current UPB (unpaid principal balance)

| 1 | Credit Score | FICO score | 812 |- Delinquency status

| 2 | First Payment Date | YYYYMM format | 202403 |- Loan age

| 8 | Occupancy Status | P=Primary, S=Second, I=Investment | P |- Months to maturity

| 11 | Original UPB | Original loan amount | 246000 |- Modification flag

| 12 | Original LTV | Loan-to-value ratio | 38 |- Zero balance code (payoff/default)

| 16 | Product Type | FRM=Fixed, ARM=Adjustable | FRM |- Foreclosure date

| 17 | Property State | 2-letter code | IL |

| 20 | **Loan Sequence Number** | Unique ID | F24Q10000019 |### Key Fields Mapping

| 21 | Loan Purpose | P=Purchase, C=Refi | N |

| IRBStudio Field | Fannie Mae Field | File | Description |

**Sample Line:**|-----------------|------------------|------|-------------|

```| loan_id | LOAN_SEQUENCE_NUMBER | Both | Unique identifier |

812|202403|N|205402|16984|000|1|P|38|23|246000|38|7.375|R|N|FRM|IL|SF|60600|F24Q10000019|N|360|02|Other sellers|Other servicers|||9||2|N|7| exposure | CURRENT_ACTUAL_UPB | Servicing | Current balance |

```| date | MONTHLY_REPORTING_PERIOD | Servicing | Reporting date |

| default_flag | CURRENT_LOAN_DELINQUENCY_STATUS | Servicing | 0-2 = current, 3+ = default |

#### 2. Performance Files (`sample_svcg_YYYY.txt`)| ltv | ORIGINAL_LTV | Origination | Loan-to-value at origination |

| score | CREDIT_SCORE | Origination | FICO score |

Contains 31 pipe-delimited columns with monthly performance updates:

### Challenge: No Explicit PD or Rating

| Position | Field | Description | Example |

|----------|-------|-------------|---------|Fannie Mae data doesn't include:

| 1 | **Loan Sequence Number** | Matches origination file | F24Q10000019 |- ❌ PD (Probability of Default) values

| 2 | **Monthly Reporting Period** | YYYYMM format | 202402 |- ❌ Internal rating grades

| 3 | **Current Actual UPB** | Current balance | 246000.00 |- ❌ Credit scores in 0-1 scale

| 4 | **Delinquency Status** | 0=Current, 1=30DPD, 2=60DPD, 3+=90+DPD | 0 |

| 5 | Loan Age | Months since origination | 000 |**Solution**: We'll derive these from available data:

| 9 | Zero Balance Code | 01=Prepaid, 03=Foreclosure, 09=REO | |1. **PD**: Estimate from delinquency status and loan characteristics

2. **Rating**: Bin loans by credit score

**Sample Line:**3. **Score**: Normalize FICO (300-850) to 0-1 scale

```

F24Q10000019|202402|246000.00|0|000|360|||||7.375|0.00||||||||||||||40||||||246000.00---

```

## Data Preparation

### IRBStudio Field Mapping

### Step 1: Load Raw Fannie Mae Data

| IRBStudio Field | Freddie Mac Source | File | Notes |

|-----------------|-------------------|------|-------|First, let's load the raw data files:

| `loan_id` | Loan Sequence Number | Both | Join key |

| `balance` | Current Actual UPB | Performance | Current exposure |```python

| `reporting_date` | Monthly Reporting Period | Performance | Parse YYYYMM |import pandas as pd

| `default_flag` | Delinquency Status ≥ 3 | Performance | 90+ DPD = default |import numpy as np

| `score` | Credit Score (normalized) | Origination | Convert to 0-1 scale |from datetime import datetime

| `rating` | Derived from Credit Score | Origination | Bin FICO into grades |

| `pd` | **Estimated** | Derived | From rating + LTV + delinquency |# Fannie Mae origination file columns (sample)

| `ltv` | Original LTV | Origination | Risk factor |orig_columns = [

    'loan_sequence_number', 'credit_score', 'first_payment_date',

### Key Challenge: Missing PD and Rating    'first_time_homebuyer_flag', 'maturity_date', 'msa',

    'mi_percentage', 'number_of_units', 'occupancy_status',

Freddie Mac data does NOT include:    'original_cltv', 'original_dti', 'original_upb',

- ❌ PD (Probability of Default) values    'original_ltv', 'original_interest_rate', 'channel',

- ❌ Internal rating grades    'prepayment_penalty_flag', 'product_type', 'property_state',

- ❌ Normalized scores (only raw FICO 300-850)    'property_type', 'postal_code', 'loan_sequence_number_dup',

    'loan_purpose', 'original_loan_term', 'number_of_borrowers',

**Solution**: We derive these using:    'seller_name', 'servicer_name', 'super_conforming_flag'

```python]

# 1. Normalize FICO to 0-1 scale

normalized_score = 1 - ((fico - 300) / 550)# Load origination data

orig_df = pd.read_csv(

# 2. Bin into rating grades    'data/FM/sample_orig_2024.txt',

if fico >= 780: rating = 'AAA'    sep='|',

elif fico >= 740: rating = 'AA'    names=orig_columns,

elif fico >= 700: rating = 'A'    header=None

# ... etc)



# 3. Estimate PD from rating + risk factors# Servicing file columns (sample)

base_pd = rating_pd_map[rating]  # e.g., 'A' = 0.005svcg_columns = [

pd = base_pd * ltv_adjustment * delinquency_multiplier    'loan_sequence_number', 'monthly_reporting_period',

```    'current_actual_upb', 'current_loan_delinquency_status',

    'loan_age', 'remaining_months_to_maturity',

---    'repurchase_flag', 'modification_flag', 'zero_balance_code',

    'zero_balance_effective_date', 'current_interest_rate',

## Quick Start: Using Pre-Prepared Data    'current_deferred_upb', 'due_date_of_last_paid_installment',

    'mi_recoveries', 'net_sales_proceeds', 'non_mi_recoveries',

IRBStudio includes pre-prepared Freddie Mac sample data. This is the fastest way to get started:    'expenses', 'legal_costs', 'maintenance_costs',

    'taxes_and_insurance', 'miscellaneous_expenses',

### Option 1: Use Existing Examples    'actual_loss_calculation', 'modification_cost'

]

```python

# The project includes ready-to-use examples# Load servicing data

cd examplessvcg_df = pd.read_csv(

python freddie_mac_dashboard_example.py    'data/FM/sample_svcg_2024.txt',

```    sep='|',

    names=svcg_columns,

This script:    header=None

1. Loads pre-prepared `data/sample_portfolio_data_fm.csv`)

2. Runs AIRB scenario comparison

3. Generates interactive dashboardprint(f"Origination records: {len(orig_df):,}")

4. Outputs capital impact analysisprint(f"Servicing records: {len(svcg_df):,}")

```

### Option 2: Load Sample Data Directly

### Step 2: Merge and Transform Data

```python

from irbstudio import run_scenario_comparison```python

# Merge origination and servicing data

# Use the included sample datasetportfolio = svcg_df.merge(

results = run_scenario_comparison(    orig_df,

    config_path="examples/sample_config.yaml",    on='loan_sequence_number',

    portfolio_path="data/sample_portfolio_data_fm.csv",    how='inner'

    n_iterations=1000,)

    random_seed=42,

    output_dir="results/freddie_mac_quick_start"# Parse dates

)portfolio['monthly_reporting_period'] = pd.to_datetime(

    portfolio['monthly_reporting_period'],

# View results    format='%m/%d/%Y'

print(f"Mean AIRB RWA: ${results['Baseline']['AIRB']['mean']:,.0f}"))

print(f"Dashboard: results/freddie_mac_quick_start/scenario_comparison_dashboard.html")

```# Create derived fields

portfolio['normalized_score'] = 1 - (

---    (portfolio['credit_score'] - 300) / 550

)

## Data Preparation from Raw Files

# Map delinquency to default flag

If you want to prepare data from raw Freddie Mac files, follow these steps:# 0 = current, 1 = 30 days, 2 = 60 days, 3+ = default

portfolio['default_flag'] = (

### Step 1: Download Freddie Mac Data    portfolio['current_loan_delinquency_status'] >= 3

).astype(int)

1. Register at https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

2. Download quarterly files (e.g., 2024 Q1, Q2, Q3, Q4)# Create into_default_flag (new defaults this period)

3. Extract to `data/FM/` directory:portfolio = portfolio.sort_values(['loan_sequence_number', 'monthly_reporting_period'])

   - `sample_orig_2024.txt` (origination data)portfolio['prev_default'] = portfolio.groupby('loan_sequence_number')['default_flag'].shift(1).fillna(0)

   - `sample_svcg_2024.txt` (performance data)portfolio['into_default_flag'] = (

    (portfolio['default_flag'] == 1) & (portfolio['prev_default'] == 0)

### Step 2: Load and Merge Data).astype(int)



```python# Create rating grades based on credit score

import pandas as pddef assign_rating(score):

import numpy as np    if score >= 780: return 'AAA'

    elif score >= 740: return 'AA'

# Define column names (Freddie Mac files have no headers)    elif score >= 700: return 'A'

orig_columns = [    elif score >= 660: return 'BBB'

    'credit_score', 'first_payment_date', 'first_time_homebuyer_flag',    elif score >= 620: return 'BB'

    'maturity_date', 'msa', 'mi_percentage', 'number_of_units',    elif score >= 580: return 'B'

    'occupancy_status', 'original_cltv', 'original_dti', 'original_upb',    else: return 'CCC'

    'original_ltv', 'original_interest_rate', 'channel',

    'prepayment_penalty_flag', 'product_type', 'property_state',portfolio['rating'] = portfolio['credit_score'].apply(assign_rating)

    'property_type', 'postal_code', 'loan_sequence_number',

    'loan_purpose', 'original_loan_term', 'number_of_borrowers',# Estimate PD from delinquency and score

    'seller_name', 'servicer_name', 'super_conforming_flag',# Simple heuristic: use historical default rates by rating

    'pre_harp_loan_sequence_number'rating_pd = {

]    'AAA': 0.0005, 'AA': 0.001, 'A': 0.005,

    'BBB': 0.01, 'BB': 0.03, 'B': 0.05, 'CCC': 0.10

perf_columns = [}

    'loan_sequence_number', 'monthly_reporting_period',portfolio['pd'] = portfolio['rating'].map(rating_pd)

    'current_actual_upb', 'current_loan_delinquency_status',

    'loan_age', 'remaining_months_to_maturity',# Add some noise to PD based on individual characteristics

    'repurchase_flag', 'modification_flag', 'zero_balance_code',portfolio['pd'] = portfolio['pd'] * (

    'zero_balance_effective_date', 'current_interest_rate',    1 + 0.2 * (portfolio['original_ltv'] - 80) / 20

    'current_deferred_upb', 'due_date_of_last_paid_installment',)  # Adjust for LTV

    'mi_recoveries', 'net_sales_proceeds', 'non_mi_recoveries',portfolio['pd'] = portfolio['pd'].clip(0.0001, 0.50)  # Reasonable bounds

    'expenses', 'legal_costs', 'maintenance_costs',

    'taxes_and_insurance', 'miscellaneous_expenses',print("\nTransformed Portfolio Sample:")

    'actual_loss_calculation', 'modification_cost',print(portfolio[['loan_sequence_number', 'monthly_reporting_period', 

    'step_modification_flag', 'deferred_payment_plan',               'current_actual_upb', 'credit_score', 'rating', 

    'estimated_ltv', 'zero_balance_removal_upb',               'pd', 'normalized_score', 'default_flag']].head())

    'delinquent_accrued_interest', 'delinquency_due_to_disaster',```

    'borrower_assistance_status_code', 'current_month_modification_cost'

]### Step 3: Prepare IRBStudio Format



# Load data```python

orig_df = pd.read_csv('data/FM/sample_orig_2024.txt', sep='|', names=orig_columns, header=None)# Select relevant columns and rename

perf_df = pd.read_csv('data/FM/sample_svcg_2024.txt', sep='|', names=perf_columns, header=None)irbstudio_portfolio = portfolio[[

    'loan_sequence_number',

# Merge on loan_sequence_number    'current_actual_upb',

portfolio = perf_df.merge(orig_df, on='loan_sequence_number', how='inner')    'pd',

print(f"Merged {len(portfolio):,} records")    'normalized_score',

```    'rating',

    'monthly_reporting_period',

### Step 3: Transform to IRBStudio Format    'default_flag',

    'into_default_flag',

```python    'original_ltv'

# Parse dates (YYYYMM format)]].rename(columns={

portfolio['reporting_date'] = pd.to_datetime(    'loan_sequence_number': 'loan_id',

    portfolio['monthly_reporting_period'].astype(str),    'current_actual_upb': 'balance',

    format='%Y%m',    'normalized_score': 'score',

    errors='coerce'    'monthly_reporting_period': 'reporting_date',

)    'original_ltv': 'ltv'

})

# Normalize credit score (0 = best, 1 = worst)

portfolio['credit_score'] = pd.to_numeric(portfolio['credit_score'], errors='coerce')# Filter to recent data (e.g., last 12 months)

portfolio['normalized_score'] = 1 - ((portfolio['credit_score'] - 300) / 550)cutoff_date = irbstudio_portfolio['reporting_date'].max() - pd.DateOffset(months=12)

portfolio['normalized_score'] = portfolio['normalized_score'].clip(0, 1)irbstudio_portfolio = irbstudio_portfolio[

    irbstudio_portfolio['reporting_date'] >= cutoff_date

# Assign rating grades]

def assign_rating(fico):

    if pd.isna(fico): return 'B'# Remove loans with missing critical data

    if fico >= 780: return 'AAA'irbstudio_portfolio = irbstudio_portfolio.dropna(

    elif fico >= 740: return 'AA'    subset=['balance', 'pd', 'score', 'rating']

    elif fico >= 700: return 'A')

    elif fico >= 660: return 'BBB'

    elif fico >= 620: return 'BB'# Save to CSV

    elif fico >= 580: return 'B'irbstudio_portfolio.to_csv(

    else: return 'CCC'    'data/fannie_mae_portfolio_prepared.csv',

    index=False

portfolio['rating'] = portfolio['credit_score'].apply(assign_rating))



# Estimate PDprint(f"\nPrepared portfolio: {len(irbstudio_portfolio):,} records")

rating_pd = {'AAA': 0.0005, 'AA': 0.001, 'A': 0.005, 'BBB': 0.01, print(f"Date range: {irbstudio_portfolio['reporting_date'].min()} to {irbstudio_portfolio['reporting_date'].max()}")

             'BB': 0.03, 'B': 0.05, 'CCC': 0.10}print(f"Total exposure: ${irbstudio_portfolio['balance'].sum():,.0f}")

portfolio['pd'] = portfolio['rating'].map(rating_pd)```



# Adjust for LTV---

portfolio['original_ltv'] = pd.to_numeric(portfolio['original_ltv'], errors='coerce')

ltv_adj = 1 + 0.3 * (portfolio['original_ltv'] - 80) / 20## Configuration Setup

portfolio['pd'] = portfolio['pd'] * ltv_adj.fillna(1.0)

### Create Configuration for Fannie Mae Analysis

# Adjust for current delinquency

portfolio['delinq'] = portfolio['current_loan_delinquency_status'].fillna('0').astype(str).str.strip()```yaml

delinq_mult = {'0': 1.0, '1': 2.0, '2': 4.0, '3': 8.0}# fannie_mae_config.yaml

portfolio['pd'] = portfolio['pd'] * portfolio['delinq'].apply(lambda x: delinq_mult.get(x, 1.0))

portfolio['pd'] = portfolio['pd'].clip(0.0001, 0.50)# Map prepared data to IRBStudio canonical fields

column_mapping:

# Create default flags  loan_id: loan_id

portfolio['default_flag'] = portfolio['delinq'].apply(lambda x: 1 if x not in ['0','1','2',''] else 0)  exposure: balance

portfolio = portfolio.sort_values(['loan_sequence_number', 'reporting_date'])  pd: pd

portfolio['prev_default'] = portfolio.groupby('loan_sequence_number')['default_flag'].shift(1).fillna(0)  score: score

portfolio['into_default_flag'] = ((portfolio['default_flag'] == 1) & (portfolio['prev_default'] == 0)).astype(int)  rating: rating

  date: reporting_date

# Select and rename columns  default_flag: default_flag

irbstudio_portfolio = portfolio[[  into_default_flag: into_default_flag

    'loan_sequence_number', 'current_actual_upb', 'pd', 'normalized_score',  ltv: ltv

    'rating', 'reporting_date', 'default_flag', 'into_default_flag', 'original_ltv'

]].rename(columns={# Regulatory parameters for US mortgage portfolio

    'loan_sequence_number': 'loan_id',regulatory:

    'current_actual_upb': 'balance',  jurisdiction: us

    'normalized_score': 'score',  asset_correlation: 0.15  # Basel standard for mortgage

    'original_ltv': 'ltv'  confidence_level: 0.999  # 99.9% confidence

})

# Scenarios to analyze

# Clean datascenarios:

irbstudio_portfolio['balance'] = pd.to_numeric(irbstudio_portfolio['balance'], errors='coerce')  # Current State: Baseline with current model

irbstudio_portfolio = irbstudio_portfolio[irbstudio_portfolio['balance'] > 0]  - name: "Current Model"

irbstudio_portfolio = irbstudio_portfolio.dropna(subset=['balance', 'pd', 'score', 'rating'])    description: "Current PD model performance (AUC ~0.72 typical for FICO-based)"

    pd_auc: 0.72

# Filter to recent 12 months    portfolio_default_rate: 0.015  # 1.5% typical for Fannie Mae

max_date = irbstudio_portfolio['reporting_date'].max()    lgd: 0.25  # 25% LGD for first-lien mortgages

cutoff_date = max_date - pd.DateOffset(months=12)    new_loan_rate: 0.08  # 8% new originations monthly

irbstudio_portfolio = irbstudio_portfolio[irbstudio_portfolio['reporting_date'] >= cutoff_date]    rating_pd_map:

      AAA: 0.0005

# Save      AA: 0.001

irbstudio_portfolio.to_csv('data/freddie_mac_prepared.csv', index=False)      A: 0.005

print(f"Saved {len(irbstudio_portfolio):,} records to data/freddie_mac_prepared.csv")      BBB: 0.01

```      BB: 0.03

      B: 0.05

---      CCC: 0.10



## Configuration Setup  # Improved Model: Better discrimination with additional data

  - name: "Enhanced Model"

Create `freddie_mac_config.yaml`:    description: "Improved model with payment history, DTI, property data (AUC ~0.78)"

    pd_auc: 0.78

```yaml    portfolio_default_rate: 0.015

# Column mapping    lgd: 0.25

column_mapping:    new_loan_rate: 0.08

  loan_id: loan_id    rating_pd_map:

  exposure: balance      AAA: 0.0005

  pd: pd      AA: 0.001

  score: score      A: 0.005

  rating: rating      BBB: 0.01

  date: reporting_date      BB: 0.03

  default_flag: default_flag      B: 0.05

  into_default_flag: into_default_flag      CCC: 0.10

  ltv: ltv

  # Stress Scenario: Economic downturn

# Regulatory parameters  - name: "Stress Scenario"

regulatory:    description: "Recession scenario with increased defaults and lower discrimination"

  jurisdiction: us    pd_auc: 0.68  # Model discrimination degrades in stress

  asset_correlation: 0.15    portfolio_default_rate: 0.04  # 4% default rate (stress)

  confidence_level: 0.999    lgd: 0.35  # Higher LGD in stress (longer foreclosures, lower recoveries)

    new_loan_rate: 0.03  # Lower originations in stress

# Scenarios    rating_pd_map:

scenarios:      AAA: 0.001

  - name: "Current Model"      AA: 0.002

    description: "Baseline FICO-based model (AUC ~0.72)"      A: 0.01

    pd_auc: 0.72      BBB: 0.02

    portfolio_default_rate: 0.015      BB: 0.06

    lgd: 0.25      B: 0.10

    new_loan_rate: 0.08      CCC: 0.20

    rating_pd_map:```

      AAA: 0.0005

      AA: 0.001---

      A: 0.005

      BBB: 0.01## Running the Analysis

      BB: 0.03

      B: 0.05### Example 1: Basic Scenario Comparison

      CCC: 0.10

```python

  - name: "Enhanced Model"from irbstudio import run_scenario_comparison

    description: "Improved model with additional data (AUC ~0.78)"

    pd_auc: 0.78# Run comparison of current vs. enhanced model

    portfolio_default_rate: 0.015results = run_scenario_comparison(

    lgd: 0.25    config_path="fannie_mae_config.yaml",

    new_loan_rate: 0.08    portfolio_path="data/fannie_mae_portfolio_prepared.csv",

    rating_pd_map:    n_iterations=5000,  # Higher iterations for business case

      AAA: 0.0005    random_seed=42,

      AA: 0.001    output_dir="results/fannie_mae_analysis"

      A: 0.005)

      BBB: 0.01

      BB: 0.03# Print summary

      B: 0.05print("\n" + "="*60)

      CCC: 0.10print("FANNIE MAE PORTFOLIO: CAPITAL IMPACT ANALYSIS")

print("="*60)

  - name: "Stress Scenario"

    description: "Recession with higher defaults"for scenario_name in ['Current Model', 'Enhanced Model', 'Stress Scenario']:

    pd_auc: 0.68    if scenario_name in results:

    portfolio_default_rate: 0.04        airb_stats = results[scenario_name]['AIRB']

    lgd: 0.35        print(f"\n{scenario_name}:")

    new_loan_rate: 0.03        print(f"  Mean RWA:    ${airb_stats['mean']:>15,.0f}")

    rating_pd_map:        print(f"  Std Dev:     ${airb_stats['std']:>15,.0f}")

      AAA: 0.001        print(f"  P95 RWA:     ${airb_stats['percentiles']['P95']:>15,.0f}")

      AA: 0.002

      A: 0.01# Capital savings from model improvement

      BBB: 0.02if 'capital_delta' in results:

      BB: 0.06    print(f"\nCapital Savings (Current → Enhanced): ${results['capital_delta']:>15,.0f}")

      B: 0.10    print(f"                                        (RWA reduction × 8% capital ratio)")

      CCC: 0.20

```print("\n" + "="*60)

print(f"Dashboard: results/fannie_mae_analysis/scenario_comparison_dashboard.html")

---print("="*60)

```

## Running the Analysis

### Example 2: Detailed AIRB vs SA Comparison

### Example 1: Scenario Comparison

```python

```pythonfrom irbstudio import load_config

from irbstudio import run_scenario_comparisonfrom irbstudio.data.loader import load_portfolio

from irbstudio.simulation.portfolio_simulator import PortfolioSimulator

results = run_scenario_comparison(from irbstudio.engine.integrated_analysis import IntegratedAnalysis

    config_path="freddie_mac_config.yaml",from irbstudio.engine.mortgage import AIRBMortgageCalculator, SAMortgageCalculator

    portfolio_path="data/freddie_mac_prepared.csv",

    n_iterations=5000,# Load data

    random_seed=42,config = load_config("fannie_mae_config.yaml")

    output_dir="results/freddie_mac_analysis"portfolio_df = load_portfolio(

)    "data/fannie_mae_portfolio_prepared.csv",

    config.column_mapping

# Print results)

print("\n" + "="*70)

print("FREDDIE MAC PORTFOLIO ANALYSIS")# Create analysis engine

print("="*70)analysis = IntegratedAnalysis()



for scenario in ['Current Model', 'Enhanced Model', 'Stress Scenario']:# Add both calculators

    if scenario in results:analysis.add_calculator('AIRB', AIRBMortgageCalculator(

        stats = results[scenario]['AIRB']    regulatory_params={

        print(f"\n{scenario}:")        'lgd': 0.25,

        print(f"  Mean RWA:  ${stats['mean']:>15,.0f}")        'asset_correlation': 0.15,

        print(f"  P95 RWA:   ${stats['percentiles']['P95']:>15,.0f}")        'confidence_level': 0.999

    }

if 'capital_delta' in results:))

    print(f"\nCapital Savings: ${results['capital_delta']:>15,.0f}")analysis.add_calculator('SA', SAMortgageCalculator())



print(f"\nDashboard: results/freddie_mac_analysis/scenario_comparison_dashboard.html")# Create simulator for current model

```current_scenario = config.scenarios[0]  # "Current Model"

simulator = PortfolioSimulator(

### Example 2: AIRB vs SA    portfolio_df=portfolio_df,

    score_to_rating_bounds={

```python        'AAA': (0.00, 0.02),

from irbstudio import load_config        'AA': (0.02, 0.05),

from irbstudio.data.loader import load_portfolio        'A': (0.05, 0.10),

from irbstudio.engine.integrated_analysis import IntegratedAnalysis        'BBB': (0.10, 0.20),

from irbstudio.engine.mortgage import AIRBMortgageCalculator, SAMortgageCalculator        'BB': (0.20, 0.30),

from irbstudio.simulation.portfolio_simulator import PortfolioSimulator        'B': (0.30, 0.50),

        'CCC': (0.50, 1.00)

# Load    },

config = load_config("freddie_mac_config.yaml")    rating_col='rating',

portfolio_df = load_portfolio("data/freddie_mac_prepared.csv", config.column_mapping)    loan_id_col='loan_id',

    date_col='reporting_date',

# Setup    default_col='default_flag',

analysis = IntegratedAnalysis()    into_default_flag_col='into_default_flag',

analysis.add_calculator('AIRB', AIRBMortgageCalculator({'lgd': 0.25}))    score_col='score',

analysis.add_calculator('SA', SAMortgageCalculator())    target_auc=current_scenario.pd_auc

)

simulator = PortfolioSimulator(portfolio_df=portfolio_df, target_auc=0.72)

analysis.add_scenario('Baseline', simulator, n_iterations=2000)# Add scenario

analysis.add_scenario('Current Model', simulator, n_iterations=2000)

# Run

results = analysis.run_scenario('Baseline', random_seed=42, application_start_date='2024-01-01')# Run with both calculators

results = analysis.run_scenario(

# Compare    scenario_name='Current Model',

airb = results['AIRB']['mean']    random_seed=42,

sa = results['SA']['mean']    application_start_date='2024-01-01'

print(f"AIRB RWA: ${airb:,.0f}"))

print(f"SA RWA: ${sa:,.0f}")

print(f"Savings: ${(sa - airb) * 0.08:,.0f}")# Compare AIRB vs SA

```airb_mean = results['AIRB']['mean']

sa_mean = results['SA']['mean']

---savings = (sa_mean - airb_mean) * 0.08



## Interpreting Resultsprint("\n" + "="*60)

print("AIRB vs STANDARDIZED APPROACH COMPARISON")

### Key Metricsprint("="*60)

print(f"Standardized Approach RWA:  ${sa_mean:>15,.0f}")

**Mean RWA**: Average capital requirement  print(f"AIRB Approach RWA:          ${airb_mean:>15,.0f}")

**P95 RWA**: Stress scenario (95th percentile)  print(f"RWA Reduction:              ${sa_mean - airb_mean:>15,.0f}")

**Standard Deviation**: Model uncertainty  print(f"Capital Savings:            ${savings:>15,.0f}")

**Capital Savings**: (RWA reduction) × 8%print(f"Percentage Reduction:       {((sa_mean - airb_mean) / sa_mean * 100):>15.1f}%")

print("="*60)

### Business Case Example```



```---

Current Model:  $287M RWA → $23.0M capital

Enhanced Model: $251M RWA → $20.1M capital## Interpreting Results

────────────────────────────────────────────

Savings:        $36M RWA  → $2.9M capital freed### Understanding Output Metrics



Model Cost:     $500K one-time#### 1. Mean RWA

ROI:            ($2.9M - $0.5M) / $0.5M = 480%- **Definition**: Average RWA across all Monte Carlo simulations

Payback:        2.1 months- **Use**: Primary metric for capital planning and budgeting

```- **Typical Values** (for $1B portfolio):

  - Conservative: $400M - $600M (40-60% RWA density)

---  - Moderate: $250M - $400M (25-40%)

  - Aggressive: $150M - $250M (15-25%)

## Advanced Examples

#### 2. Standard Deviation

### Vintage Analysis- **Definition**: Measure of RWA volatility/uncertainty

- **Use**: Understanding model risk

```python- **Interpretation**:

# Group by origination year  - Low (< 5% of mean): Stable, low uncertainty

for year in [2021, 2022, 2023, 2024]:  - Medium (5-15%): Typical for established portfolios

    vintage_df = portfolio_df[portfolio_df['origination_year'] == year]  - High (> 15%): High uncertainty, consider more data or model validation

    simulator = PortfolioSimulator(portfolio_df=vintage_df, target_auc=0.75)

    analysis.add_scenario(f'Vintage_{year}', simulator, n_iterations=500)#### 3. P95 RWA

    results = analysis.run_scenario(f'Vintage_{year}', random_seed=42)- **Definition**: 95th percentile (only 5% chance of exceeding)

    print(f"{year}: ${results['AIRB']['mean']:,.0f}")- **Use**: Stress testing and capital buffer determination

```- **Regulatory**: Often used for ICAAP stress scenarios



### Geographic Segmentation### Dashboard Visualization Guide



```pythonThe generated dashboard includes:

# Analyze by state

for state in ['CA', 'TX', 'FL', 'NY']:**1. Distribution Plots**

    state_df = portfolio_df[portfolio_df['property_state'] == state]- **Histogram**: Shows RWA frequency distribution

    # ... run analysis per state- **KDE Overlay**: Smooth probability density

```- **Percentile Lines**: P5, P50, P95 markers

- **Interpretation**: Look for skewness, multiple modes

---

**2. Scenario Comparison**

## Summary- **Bar Chart**: Side-by-side mean RWA comparison

- **Error Bars**: Show ±1 standard deviation

### Key Takeaways- **Use**: Quick visual comparison of scenarios



✅ **Freddie Mac is IRBStudio's primary test dataset**  **3. Summary Statistics Table**

✅ **Pipe-delimited format requires explicit column definitions**  | Metric | Current Model | Enhanced Model | Stress |

✅ **PD and ratings must be derived (not provided)**  |--------|---------------|----------------|--------|

✅ **YYYYMM date format needs parsing**  | Mean | $325M | $298M | $487M |

✅ **Rich loan characteristics enable advanced segmentation**| Std Dev | $18M | $16M | $34M |

| P95 | $356M | $325M | $545M |

### Best Practices| Skew | 0.15 | 0.12 | 0.28 |



1. ✅ Filter to recent 12-24 months for current portfolio**4. Waterfall Chart**

2. ✅ Validate FICO distribution (expect 700-750 average for prime)- Shows component-by-component RWA breakdown

3. ✅ Check for zero balance loans (exclude from analysis)- Useful for understanding drivers of change

4. ✅ Calibrate PD adjustments to your institution's experience

5. ✅ Document all assumptions for audit trail### Business Insights



### Resources#### Model Improvement Business Case



- **Freddie Mac**: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset```

- **IRBStudio Examples**: `examples/freddie_mac_dashboard_example.py`Current Model:     Mean RWA = $325M

- **Sample Data**: `data/FM/sample_orig_2024.txt`, `data/FM/sample_svcg_2024.txt`Enhanced Model:    Mean RWA = $298M

- **User Guide**: [docs/user_guide.md](user_guide.md)RWA Reduction:     $27M

- **API Reference**: [docs/api_reference.md](api_reference.md)Capital Savings:   $27M × 8% = $2.16M



---If model development cost = $500K

ROI = ($2.16M - $0.5M) / $0.5M = 332%

*This tutorial uses Freddie Mac data structure as demonstrated throughout IRBStudio's development. Always validate results with your model validation team before business decisions.*Payback period = 0.3 years (~4 months)

```

*Last Updated: October 2025*

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
