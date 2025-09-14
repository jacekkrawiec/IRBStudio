import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from pandas.tseries.offsets import DateOffset
from scipy.stats import norm, beta

from irbstudio.simulation.distribution import BetaMixtureFitter
from irbstudio.simulation.migration import calculate_migration_matrix
from irbstudio.simulation.score_generation import (
    generate_calibrated_scores,
    find_auc_calibration_factor,
)

def simulate_portfolio(
    portfolio_df: pd.DataFrame,
    score_to_rating_bounds: Dict[str, tuple],
    rating_col: str,
    loan_id_col: str,
    date_col: str,
    default_col: str,
    into_default_flag_col: str,
    score_col: str,
    application_start_date: Optional[datetime] = None,
    asset_correlation: float = 0.15,
    exposure_col: Optional[str] = None,
    target_auc: Optional[float] = None,
) -> pd.DataFrame:
    """
    Orchestrates the procedurally-faithful portfolio simulation.

    This function performs a hybrid simulation by:
    1. Segmenting the portfolio into historical and application samples.
    2. Calculating a long-term average PD from the historical data.
    3. Simulating rating migrations for existing clients.
    4. Drawing scores for new clients from a fitted distribution.
    5. Combining the segments and applying the final PD calibration.

    Args:
        portfolio_df (pd.DataFrame): The full portfolio dataset, including history.
        score_to_rating_bounds (Dict[str, tuple]): Dict mapping rating to (min_score, max_score).
        rating_col (str): The name of the column containing rating grades.
        loan_id_col (str): The name of the column containing unique loan identifiers.
        date_col (str): The name of the column containing the snapshot date.
        default_col (str): The name of the column containing the default flag.
        into_default_flag_col (str): The name of the column flagging the period of default entry.
        score_col (str): The name of the column containing the current model's score.
        application_start_date (Optional[datetime], optional): The date from which
            the application sample begins. If not provided, it defaults to a
            12-month window ending on the most recent date in the dataset.
            Defaults to None.
        asset_correlation (float, optional): The asset correlation parameter (R)
            used in the Merton model. Defaults to 0.15, a common value for
            mortgages.
        exposure_col (Optional[str], optional): The name of the exposure column.
            If provided, it will be carried through the simulation. Defaults to None.
        target_auc (Optional[float], optional): The target AUC for new client scores.
            If provided, new client scores will be calibrated to achieve this AUC
            while preserving the fitted distribution shape. Defaults to None.

    Returns:
        pd.DataFrame: The simulated portfolio for the reporting date with new
                      ratings and PDs.
    """
    # Ensure date column is in datetime format for comparison
    portfolio_df[date_col] = pd.to_datetime(portfolio_df[date_col])

    # --- 1. Data Segmentation ---
    if application_start_date is None:
        # If no start date is provided, automatically determine the 12-month window
        most_recent_date = portfolio_df[date_col].max()
        application_start_date = most_recent_date - DateOffset(months=11)
        application_start_date = application_start_date.replace(day=1)

    # First, separate out all defaulted exposures to add back at the end
    defaulted_df = portfolio_df[portfolio_df[default_col] == 1].copy()
    
    # Create clean dataset without defaulted observations
    clean_portfolio_df = portfolio_df.loc[~portfolio_df.index.isin(defaulted_df.index)].copy()
    
    # Split the clean portfolio into historical and application samples
    historical_df = clean_portfolio_df[clean_portfolio_df[date_col] < application_start_date]
    application_df = clean_portfolio_df[
        clean_portfolio_df[date_col] >= application_start_date
    ].copy()

    if historical_df.empty:
        raise ValueError("Historical data is empty. Cannot proceed with simulation.")
    if application_df.empty:
        # It's a valid scenario to have no application data (e.g., only running on history)
        # We can log this and return an empty DataFrame or handle as needed.
        # For now, we'll raise an error to make the behavior explicit.
        raise ValueError(
            f"No application data found from date {application_start_date} onwards. "
            "Cannot proceed with simulation."
        )

    # --- 2. Calculate Long-Term PD from Historical Data ---
    # Calculate monthly default rates per rating
    monthly_rates = historical_df.groupby([date_col, rating_col]).agg(
        defaults=(into_default_flag_col, 'sum'),
        total=(loan_id_col, 'count')
    ).reset_index()

    # Avoid division by zero
    monthly_rates['monthly_dr'] = 0.0
    monthly_rates.loc[monthly_rates['total'] > 0, 'monthly_dr'] = monthly_rates['defaults'] / monthly_rates['total']

    # Calculate long-term average default rate per rating
    rating_pd_map = monthly_rates.groupby(rating_col)['monthly_dr'].mean().to_dict()

    # Add PD for 'D' rating, ensuring it's present
    rating_pd_map['D'] = 1.0

    # --- 3. Further Segmentation (Application Sample) ---
    historical_ids = set(historical_df[loan_id_col].unique())
    application_ids = set(application_df[loan_id_col].unique())

    existing_client_ids = historical_ids.intersection(application_ids)
    new_client_ids = application_ids - historical_ids

    existing_clients_df = application_df[
        application_df[loan_id_col].isin(existing_client_ids)
    ].copy()
    new_clients_df = application_df[
        application_df[loan_id_col].isin(new_client_ids)
    ].copy()

    # --- 4. Facility-Level Simulation for Historical Sample ---
    # 4.1. Implement facility-level default labeling
    defaulted_facility_ids = set(
        historical_df.loc[historical_df[into_default_flag_col] == 1, loan_id_col].unique()
    )
    non_defaulted_facility_ids = set(
        historical_df.loc[historical_df[into_default_flag_col] == 0, loan_id_col].unique()
    )

    num_defaulted_facilities = len(defaulted_facility_ids)
    num_non_defaulted_facilities = len(non_defaulted_facility_ids)

    # --- 4.2. Generate stable idiosyncratic scores ---
    bmf = BetaMixtureFitter(n_components=2)
    
    # Prepare data for supervised fitting
    fit_df = historical_df[[score_col, into_default_flag_col]].dropna()
    X_fit = fit_df[score_col].values
    y_fit = fit_df[into_default_flag_col].values

    try:
        bmf.fit(X_fit, y_fit)
    except Exception as e:
        print(f"Supervised fitting failed: {e}. Falling back to unsupervised.")
        # Fallback to unsupervised fitting on non-default scores
        non_default_scores = historical_df.loc[historical_df[default_col] == 0, score_col].dropna()
        if len(non_default_scores) < 10:
            synthetic_scores = np.random.beta(2, 5, size=100)
            non_default_scores = pd.Series(synthetic_scores)
        clipped_scores = non_default_scores.clip(0.001, 0.999).values
        bmf.fit(clipped_scores)

    # Generate idiosyncratic scores for each observation based on into_default_flag
    
    ######################### To be changed
    
    nd_scores_dict = dict(
        zip(
            non_defaulted_facility_ids,
            bmf.sample(n_samples=num_non_defaulted_facilities, component=0, target_auc=target_auc)
        )
    )

    d_scores_dict = dict(
        zip(
            defaulted_facility_ids,
            bmf.sample(n_samples=num_defaulted_facilities, component=1, target_auc=target_auc)
        )
    )

    historical_df['idiosyncratic_score'] = np.nan
    historical_df.loc[historical_df[into_default_flag_col] == 1, 'idiosyncratic_score'] = (
        historical_df.loc[historical_df[into_default_flag_col] == 1, loan_id_col].map(d_scores_dict)
    )

    historical_df.loc[historical_df[into_default_flag_col] == 0, 'idiosyncratic_score'] = (
        historical_df.loc[historical_df[into_default_flag_col] == 0, loan_id_col].map(nd_scores_dict)
    )

    ##########################

    # --- 4.3. Simulate final dynamic scores ---
    systemic_factor = _infer_systemic_factor(
        historical_df.copy(), date_col, rating_col, loan_id_col
    )

    # Use the already created 'idiosyncratic_score' column
    historical_df = historical_df.merge(
        systemic_factor, left_on=date_col, right_index=True, how='left'
    )
    historical_df['systemic_factor'] = historical_df['systemic_factor'].ffill().fillna(0)

    idiosyncratic_factor = norm.ppf(historical_df['idiosyncratic_score'].clip(0.001, 0.999))
    R = asset_correlation
    asset_value = np.sqrt(R) * idiosyncratic_factor + np.sqrt(1 - R) * historical_df['systemic_factor']
    historical_df['simulated_score'] = norm.cdf(asset_value)

    # --- 4.4. Map scores to ratings ---
    historical_df['simulated_rating'] = _apply_score_bounds_to_ratings(
        historical_df['simulated_score'], score_to_rating_bounds
    )

    # --- 4.5. Handle defaulted observations ---
    historical_df.loc[historical_df[default_col] == 1, 'simulated_rating'] = 'D'
    historical_df['simulated_pd'] = historical_df['simulated_rating'].map(rating_pd_map)
    historical_df.loc[historical_df[default_col] == 1, 'simulated_pd'] = 1.0

    historical_df.drop(
        columns=['idiosyncratic_score', 'systemic_factor'],
        inplace=True,
    )

    # --- 5. Handle New and Existing Clients in Application Sample ---
    simulated_migration_matrix = calculate_migration_matrix(
        historical_df,
        id_col=loan_id_col,
        date_col=date_col,
        rating_col='simulated_rating',
    )

    # --- 5.1. Handle "New Clients" ---
    if not new_clients_df.empty:
        num_new_clients = new_clients_df[loan_id_col].nunique()
        new_client_scores = bmf.sample(n_samples=num_new_clients, component=0, target_auc=target_auc)
        new_client_score_map = dict(zip(new_clients_df[loan_id_col].unique(), new_client_scores))
        new_clients_df['simulated_score'] = new_clients_df[loan_id_col].map(new_client_score_map)
        new_clients_df['simulated_rating'] = _apply_score_bounds_to_ratings(
            new_clients_df['simulated_score'], score_to_rating_bounds
        )
        new_clients_df_sorted = new_clients_df.sort_values([loan_id_col, date_col])
        new_clients_df = new_clients_df_sorted.groupby(loan_id_col, group_keys=False).apply(
            _apply_migrations,
            start_rating_col='simulated_rating',
            migration_matrix=simulated_migration_matrix,
            keep_first_rating=True
        )
    
    # --- 5.2. Handle "Existing Clients" ---
    if not existing_clients_df.empty:
        last_historical_ratings = historical_df.sort_values(date_col).groupby(loan_id_col)['simulated_rating'].last()
        
        existing_clients_df['last_historical_rating'] = existing_clients_df[loan_id_col].map(last_historical_ratings)
        
        existing_clients_df_sorted = existing_clients_df.sort_values([loan_id_col, date_col])
        
        existing_clients_df = existing_clients_df_sorted.groupby(loan_id_col, group_keys=False).apply(
            _apply_migrations,
            start_rating_col='last_historical_rating',
            migration_matrix=simulated_migration_matrix,
            keep_first_rating=False
        )

    # --- 6. Combine and Finalize ---
    simulated_portfolio_df = pd.concat([historical_df, new_clients_df, existing_clients_df], ignore_index=True)
    
    simulated_portfolio_df['simulated_pd'] = simulated_portfolio_df['simulated_rating'].map(rating_pd_map)
    
    # Ensure defaulted exposures are handled correctly in the final output
    defaulted_df['simulated_pd'] = 1.0
    defaulted_df['simulated_rating'] = 'D'
    
    simulated_portfolio_df = pd.concat([simulated_portfolio_df, defaulted_df], ignore_index=True)

    # Fill any missing PDs for ratings that might not have been in the historical map
    # (e.g., a new rating appears in simulation)
    simulated_portfolio_df['simulated_pd'] = simulated_portfolio_df['simulated_pd'].fillna(
        simulated_portfolio_df.groupby('simulated_rating')['simulated_pd'].transform('mean')
    )
    # If still missing, fill with a reasonable default
    simulated_portfolio_df['simulated_pd'].fillna(0.1, inplace=True)

    return simulated_portfolio_df

def _apply_score_bounds_to_ratings(scores: pd.Series, score_to_rating_bounds: Dict[str, tuple]) -> pd.Series:
    """
    Map scores to ratings using explicit user-provided score bounds for each rating.

    Args:
        scores: pd.Series of scores to map
        score_to_rating_bounds: Dict mapping rating to (min_score, max_score)

    Returns:
        pd.Series of ratings
    """
    # Prepare bins and labels for pd.cut
    # Sort by min_bound
    sorted_items = sorted(score_to_rating_bounds.items(), key=lambda x: x[1][0])
    labels = [item[0] for item in sorted_items]
    bounds = [item[1] for item in sorted_items]
    # Flatten bounds to get bin edges
    bin_edges = [bounds[0][0]] + [b[1] for b in bounds]
    # pd.cut is right-exclusive by default, so this matches min <= x < max
    ratings = pd.cut(scores, bins=bin_edges, labels=labels, include_lowest=True, right=False)
    # Handle edge case: if score == max of last bin, assign last label
    last_max = bin_edges[-1]
    ratings = ratings.astype(object)
    ratings[scores == last_max] = labels[-1]
    return ratings

def _infer_systemic_factor(
    historical_df: pd.DataFrame, date_col: str, rating_col: str, loan_id_col: str
) -> pd.Series:
    """
    Infers a historical systemic risk factor (M_t) from observed rating migrations.

    This function calculates the net upgrade/downgrade percentage for each period
    and uses the inverse CDF of a standard normal distribution to transform this
    into a systemic factor.

    Args:
        historical_df (pd.DataFrame): The historical portfolio data.
        date_col (str): The name of the column containing the snapshot date.
        rating_col (str): The name of the column containing rating grades.

    Returns:
        pd.Series: A series of systemic factors indexed by date.
    """
    # Handle mixed types by converting all ratings to strings for ordering
    historical_df = historical_df.copy()
    historical_df['rating_str'] = historical_df[rating_col].astype(str)
    
    # Create an ordered categorical based on natural ordering, with 'D' at the end
    unique_ratings = historical_df['rating_str'].unique()
    numeric_ratings = [r for r in unique_ratings if r.isdigit()]
    numeric_ratings = sorted(numeric_ratings, key=int)
    
    # Add non-numeric ratings at the end (typically just 'D')
    non_numeric_ratings = [r for r in unique_ratings if not r.isdigit()]
    rating_order = numeric_ratings + non_numeric_ratings
    
    historical_df['rating_cat'] = pd.Categorical(
        historical_df['rating_str'], categories=rating_order, ordered=True
    )
    historical_df['rating_code'] = historical_df['rating_cat'].cat.codes

    # Calculate rating changes period over period for each loan
    historical_df_sorted = historical_df.sort_values(by=[loan_id_col, date_col])
    historical_df_sorted['prev_rating_code'] = historical_df_sorted.groupby(loan_id_col)[
        'rating_code'
    ].shift(1)

    # Determine upgrades, downgrades, and stable ratings
    delta = (
        historical_df_sorted['rating_code'] - historical_df_sorted['prev_rating_code']
    )
    historical_df_sorted['change'] = np.select(
        [delta > 0, delta < 0], ['Downgrade', 'Upgrade'], default='Stable'
    )

    # Calculate net upgrade percentage for each period
    migrations_by_date = (
        historical_df_sorted.groupby([date_col, 'change'])
        .size()
        .unstack(fill_value=0)
    )
    migrations_by_date['total'] = migrations_by_date.sum(axis=1)
    migrations_by_date['net_upgrades'] = (
        migrations_by_date.get('Upgrade', 0) - migrations_by_date.get('Downgrade', 0)
    )
    migrations_by_date['net_upgrade_pct'] = (
        migrations_by_date['net_upgrades'] / migrations_by_date['total']
    )

    # Convert net upgrade percentage to a probability-like measure (0 to 1)
    # (net_upgrade_pct + 1) / 2 maps [-1, 1] to [0, 1]
    prob_of_upgrade = (migrations_by_date['net_upgrade_pct'] + 1) / 2
    
    # Use the inverse CDF of a standard normal distribution (probit function)
    # to get the systemic factor M_t
    from scipy.stats import norm
    systemic_factor_values = norm.ppf(prob_of_upgrade.clip(0.001, 0.999)) # Clip for stability
    systemic_factor = pd.Series(systemic_factor_values, index=prob_of_upgrade.index, name='systemic_factor')

    return systemic_factor

def _apply_migrations(
    group: pd.DataFrame,
    start_rating_col: str,
    migration_matrix: pd.DataFrame,
    keep_first_rating: bool,
) -> pd.DataFrame:
    """
    Applies rating migrations to a group of observations for a single facility.

    Args:
        group (pd.DataFrame): The observation group for a single loan.
        start_rating_col (str): The name of the column containing the initial
            rating to start the migration from.
        migration_matrix (pd.DataFrame): The matrix of migration probabilities.
        keep_first_rating (bool): If True, the first observation keeps the
            start_rating. If False, the first observation is also migrated.

    Returns:
        pd.DataFrame: The group with the 'simulated_rating' column populated.
    """
    # Get the starting rating from the first row of the group
    current_rating = group[start_rating_col].iloc[0]
    ratings = []

    if keep_first_rating:
        ratings.append(current_rating)
        num_migrations = len(group) - 1
    else:
        num_migrations = len(group)

    for _ in range(num_migrations):
        if current_rating in migration_matrix.index:
            probs = migration_matrix.loc[current_rating].values
            # Ensure probabilities sum to 1, handle potential floating point inaccuracies
            probs /= probs.sum()
            current_rating = np.random.choice(migration_matrix.columns, p=probs)
        # If rating is not in matrix (e.g., 'D' or a new rating), it stays there
        ratings.append(current_rating)

    group['simulated_rating'] = ratings
    return group
