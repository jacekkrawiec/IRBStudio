import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def _get_auc(labels, scores):
    """Helper to calculate AUC, handling cases with only one class."""
    if len(np.unique(labels)) < 2:
        return 0.5
    # For AUC, higher scores should correlate with the positive class (1)
    return roc_auc_score(labels, scores)

def generate_sample_data(num_facilities=10000, target_auc=0.7):
    """
    Generates a sample credit portfolio dataset with monthly observations.
    The 'score' column is treated as a Probability of Default (PD), so a higher
    score indicates higher risk.

    Args:
        num_facilities (int): The number of unique facilities to generate.
        target_auc (float): The target AUC for the generated scores against the
                            `into_default_flag`.

    Returns:
        pd.DataFrame: A DataFrame containing the sample portfolio data.
    """
    print(f"Generating sample data for {num_facilities} facilities with target AUC of {target_auc}...")
    print("NOTE: Score is treated as PD (higher score = higher risk).")

    # 1. Define Time Span
    start_date = pd.Timestamp('2015-01-01')
    end_date = pd.Timestamp('2025-12-31')
    date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='ME'))
    reporting_cutoff_date = end_date - pd.DateOffset(months=12)

    # 2. Generate Facility Lifecycles & Base Scores
    facility_ids = [f'FAC_{i:05d}' for i in range(num_facilities)]
    facilities_metadata = []

    # Define score distributions. Higher scores for "bad" clients (higher PD).
    good_dist_mean_initial = 0.05  # Low score for good clients
    bad_dist_mean_initial = 0.35   # High score for bad clients
    separation = bad_dist_mean_initial - good_dist_mean_initial

    # Iteratively find the right separation to hit the target AUC
    print("Calibrating score distributions to meet target AUC...")
    temp_labels = np.concatenate([np.zeros(5000), np.ones(5000)]) # 0=good, 1=bad
    current_auc = 0.0
    for i in range(15):
        # Good clients (label 0) get low scores, Bad clients (label 1) get high scores
        good_scores = np.random.beta(good_dist_mean_initial * 10, (1 - good_dist_mean_initial) * 10, 5000)
        bad_scores = np.random.beta(bad_dist_mean_initial * 10, (1 - bad_dist_mean_initial) * 10, 5000)
        current_auc = _get_auc(temp_labels, np.concatenate([good_scores, bad_scores]))

        if abs(current_auc - target_auc) < 0.005:
            break

        # Adjust separation based on AUC error
        if current_auc > 0.01 and current_auc < 0.99:
            adjustment_factor = target_auc / current_auc
            separation *= adjustment_factor
        
        center = 0.2
        separation = np.clip(separation, 0.05, 0.7)
        good_dist_mean_initial = np.clip(center - separation / 2, 0.01, 0.99)
        bad_dist_mean_initial = np.clip(center + separation / 2, 0.01, 0.99)

    print(f"Final score distributions calibrated. Estimated AUC: {current_auc:.2f}")

    for facility_id in facility_ids:
        origination_date = start_date + pd.DateOffset(days=np.random.randint(0, (end_date - start_date).days - 365))
        
        max_months = (end_date - origination_date).days // 30
        if max_months <= 24 or np.random.rand() < 0.7:
            exit_date = end_date
        else:
            exit_date = origination_date + pd.DateOffset(months=np.random.randint(24, max_months))
        exit_date = min(exit_date, end_date)

        will_default = np.random.rand() < 0.05
        default_date = None
        base_score_dist = 'good'
        if will_default:
            min_default_days = 120
            max_default_days = (exit_date - origination_date).days
            if max_default_days > min_default_days:
                default_date = origination_date + pd.DateOffset(days=np.random.randint(min_default_days, max_default_days))
                base_score_dist = 'bad'

        if base_score_dist == 'good':
            base_score = np.random.beta(good_dist_mean_initial * 10, (1 - good_dist_mean_initial) * 10)
        else:
            base_score = np.random.beta(bad_dist_mean_initial * 10, (1 - bad_dist_mean_initial) * 10)

        facilities_metadata.append({
            'facility_id': facility_id, 'origination_date': origination_date, 'exit_date': exit_date,
            'default_date': default_date, 'initial_exposure': np.random.uniform(50000, 1000000),
            'base_score': base_score
        })

    # 3. Create Monthly Observations
    all_observations = []
    for facility in tqdm(facilities_metadata, desc="Processing facilities"):
        facility_id = facility['facility_id']
        origination_date = facility['origination_date']
        exit_date = facility['exit_date']
        default_date = facility['default_date']
        initial_exposure = facility['initial_exposure']
        base_score = facility['base_score']
        amortization_rate = np.random.uniform(0.001, 0.005)
        facility_dates = date_range[(date_range >= origination_date) & (date_range <= exit_date)]
        
        exposure_at_default = None

        for i, obs_date in enumerate(facility_dates):
            is_defaulted = 1 if default_date and obs_date >= default_date else 0

            if exposure_at_default is not None:
                exposure = exposure_at_default
            else:
                exposure = initial_exposure * (1 - (i * amortization_rate))

            if is_defaulted and exposure_at_default is None:
                if i > 0:
                    exposure_at_default = initial_exposure * (1 - ((i - 1) * amortization_rate))
                else:
                    exposure_at_default = initial_exposure
                exposure = exposure_at_default

            into_default_flag = 0
            if default_date and not is_defaulted:
                if default_date <= obs_date + pd.DateOffset(months=12):
                    into_default_flag = 1
            
            if obs_date >= reporting_cutoff_date and not (is_defaulted or into_default_flag == 1):
                into_default_flag = np.nan

            if is_defaulted:
                score = 1.0
            else:
                score = np.clip(base_score + np.random.normal(0, 0.01), 0.01, 0.99)

            all_observations.append({
                'facility_id': facility_id, 'observation_date': obs_date, 'exposure': exposure,
                'default_flag': is_defaulted, 'into_default_flag': into_default_flag, 'score': score
            })
            
            if is_defaulted:
                break

    df = pd.DataFrame(all_observations)

    # 4. Add Ratings based on Score
    # Higher score = higher risk = higher (worse) rating number
    score_bins = [0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.95, 1.01]
    rating_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] # 1 is safest, 9 is riskiest
    
    df['rating'] = pd.cut(df['score'], bins=score_bins, labels=rating_labels, right=False)
    df['rating'] = df['rating'].astype(str)
    df.loc[df['default_flag'] == 1, 'rating'] = 'D'

    print("Sample data generation complete.")
    return df

if __name__ == '__main__':
    sample_df = generate_sample_data(num_facilities=10000, target_auc=0.7)
    
    output_path = './data/sample_portfolio_data.csv'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df.to_csv(output_path, index=False)
    
    print(f"Data saved to {output_path}")
    print("\n--- Data Summary ---")
    print(f"Total observations: {len(sample_df)}")
    print(f"Unique facilities: {sample_df['facility_id'].nunique()}")
    print(f"Date range: {sample_df['observation_date'].min().date()} to {sample_df['observation_date'].max().date()}")
    
    # Validate AUC
    auc_df = sample_df.dropna(subset=['into_default_flag', 'score'])
    actual_auc = _get_auc(auc_df['into_default_flag'], auc_df['score'])
    print(f"\nActual AUC of generated scores: {actual_auc:.4f}")

    print("\n--- Default Information ---")
    final_state = sample_df.loc[sample_df.groupby('facility_id')['observation_date'].idxmax()]
    print(f"Facilities in default at end of their term: {final_state['default_flag'].sum()}")
    
    print("\n--- `into_default_flag` Summary ---")
    print(sample_df['into_default_flag'].value_counts(dropna=False))

    print("\n--- Rating Distribution (latest observation) ---")
    print(final_state['rating'].value_counts().sort_index())

    print("\n--- Sample Rows ---")
    print(sample_df.head())
    
    print("\n--- Sample Rows (Defaulted) ---")
    defaulted_sample = sample_df[sample_df['default_flag'] == 1]
    if not defaulted_sample.empty:
        print(defaulted_sample.head())
    else:
        print("No defaulted facilities in this sample.")
    print("\n--- Data Summary ---")
    print(f"Total observations: {len(sample_df)}")
    print(f"Unique facilities: {sample_df['facility_id'].nunique()}")
    print(f"Date range: {sample_df['observation_date'].min().date()} to {sample_df['observation_date'].max().date()}")
    
    # Validate AUC
    auc_df = sample_df.dropna(subset=['into_default_flag', 'score'])
    actual_auc = _get_auc(auc_df['into_default_flag'], auc_df['score'])
    print(f"\nActual AUC of generated scores: {actual_auc:.4f}")

    print("\n--- Default Information ---")
    final_state = sample_df.loc[sample_df.groupby('facility_id')['observation_date'].idxmax()]
    print(f"Facilities in default at end of their term: {final_state['default_flag'].sum()}")
    
    print("\n--- `into_default_flag` Summary ---")
    print(sample_df['into_default_flag'].value_counts(dropna=False))

    print("\n--- Rating Distribution (latest observation) ---")
    print(final_state['rating'].value_counts().sort_index())

    print("\n--- Sample Rows ---")
    print(sample_df.head())
    
    print("\n--- Sample Rows (Defaulted) ---")
    defaulted_sample = sample_df[sample_df['default_flag'] == 1]
    if not defaulted_sample.empty:
        print(defaulted_sample.head())
    else:
        print("No defaulted facilities in this sample.")



