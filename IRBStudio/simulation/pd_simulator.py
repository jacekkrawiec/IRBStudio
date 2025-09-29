import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from pandas.tseries.offsets import DateOffset
from scipy.stats import norm, beta
import time

from irbstudio.simulation.distribution import BetaMixtureFitter
from irbstudio.simulation.migration import calculate_migration_matrix
from irbstudio.simulation.score_generation import (
    generate_calibrated_scores,
    find_auc_calibration_factor,
)
from irbstudio.utils.logging import get_logger

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

    logger = get_logger(__name__)
    logger.info("Starting portfolio simulation.")
    import time
    start_time = time.time()
    # Ensure date column is in datetime format for comparison
    portfolio_df = portfolio_df.copy()
    portfolio_df[date_col] = pd.to_datetime(portfolio_df[date_col])

    # Apply sorting to not repeat sorting at later stages
    portfolio_df = portfolio_df.sort_values(by=[loan_id_col, date_col])

    logger.info("Segmenting portfolio into historical and application samples.")
    if application_start_date is None:
        most_recent_date = portfolio_df[date_col].max()
        application_start_date = most_recent_date - DateOffset(months=11)
        application_start_date = application_start_date.replace(day=1)
        logger.info(f"No application_start_date provided. Using default: {application_start_date}")

    defaulted_df = portfolio_df[portfolio_df[default_col] == 1].copy()
    clean_portfolio_df = portfolio_df.loc[~portfolio_df.index.isin(defaulted_df.index)].copy()
    historical_df = clean_portfolio_df[clean_portfolio_df[date_col] < application_start_date].copy()
    application_df = clean_portfolio_df[
        clean_portfolio_df[date_col] >= application_start_date
    ].copy()

    logger.info(f"Historical sample size: {len(historical_df)}; Application sample size: {len(application_df)}")

    if historical_df.empty:
        logger.error("Historical data is empty. Cannot proceed with simulation.")
        raise ValueError("Historical data is empty. Cannot proceed with simulation.")
    if application_df.empty:
        logger.error(f"No application data found from date {application_start_date} onwards. Cannot proceed with simulation.")
        raise ValueError(
            f"No application data found from date {application_start_date} onwards. "
            "Cannot proceed with simulation."
        )

    # # --- 2. Calculate Long-Term PD from Historical Data ---
    # # Calculate monthly default rates per rating
    # logger.info("Calculating long-term average PD from historical data.")
    # monthly_rates = historical_df.groupby([date_col, rating_col]).agg(
    #     defaults=(into_default_flag_col, 'sum'),
    #     total=(loan_id_col, 'count')
    # ).reset_index()

    # monthly_rates['monthly_dr'] = 0.0
    # monthly_rates.loc[monthly_rates['total'] > 0, 'monthly_dr'] = monthly_rates['defaults'] / monthly_rates['total']
    # rating_pd_map = monthly_rates.groupby(rating_col)['monthly_dr'].mean().to_dict()
    # rating_pd_map['D'] = 1.0

    # --- 3. Further Segmentation (Application Sample) ---
    logger.info("Segmenting application sample into new and existing clients.")
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
    logger.info(f"Existing clients: {len(existing_clients_df)}, New clients: {len(new_clients_df)}")

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
    bad_proportion = num_defaulted_facilities / historical_df[loan_id_col].nunique()
    # --- 4.2. Generate stable idiosyncratic scores ---
    logger.info("Fitting Beta Mixture Model to historical scores.")
    bmf = BetaMixtureFitter(n_components=2)
    fit_df = historical_df[[score_col, into_default_flag_col]].dropna()
    X_fit = fit_df[score_col].values
    y_fit = fit_df[into_default_flag_col].values

    try:
        bmf.fit(X_fit, y_fit)
        logger.info("Supervised fitting of Beta Mixture Model succeeded.")
    except Exception as e:
        logger.warning(f"Supervised fitting failed: {e}. Falling back to unsupervised.")
        non_default_scores = historical_df.loc[historical_df[default_col] == 0, score_col].dropna()
        if len(non_default_scores) < 10:
            synthetic_scores = np.random.beta(2, 5, size=100)
            non_default_scores = pd.Series(synthetic_scores)
        clipped_scores = non_default_scores.clip(0.001, 0.999).values
        bmf.fit(clipped_scores)
        logger.info("Unsupervised fitting of Beta Mixture Model succeeded.")

    # Generate idiosyncratic scores for each observation based on into_default_flag

    logger.info("Generating idiosyncratic scores for historical sample.")
    gamma = find_auc_calibration_factor(bmf, target_auc)
    scores_good, scores_bad = generate_calibrated_scores(bmf, gamma, num_non_defaulted_facilities, num_defaulted_facilities)

    nd_scores_dict = dict(
        zip(
            non_defaulted_facility_ids,
            scores_good
        )
    )

    d_scores_dict = dict(
        zip(
            defaulted_facility_ids,
            scores_bad
        )
    )

    historical_df['idiosyncratic_score'] = np.nan
    mask_default = historical_df[into_default_flag_col] == 1
    mask_non_default = historical_df[into_default_flag_col] == 0
    historical_df.loc[mask_default, 'idiosyncratic_score'] = (
        historical_df.loc[mask_default, loan_id_col].map(d_scores_dict)
    )
    historical_df.loc[mask_non_default, 'idiosyncratic_score'] = (
        historical_df.loc[mask_non_default, loan_id_col].map(nd_scores_dict)
    )

    # --- 4.3. Simulate final dynamic scores ---
    logger.info("Inferring systemic risk factor from historical migrations.")
    systemic_factor = _infer_systemic_factor(
        historical_df, date_col, rating_col, loan_id_col, score_col
    )

    if len(systemic_factor) < len(historical_df[date_col].unique()):
        # add missing dates to the systemic factor series
        missing_dates = set(historical_df[date_col].unique()) - set(systemic_factor.index)
        # add missing dates with NaN values
        for missing_date in missing_dates:  
            systemic_factor.loc[missing_date] = np.nan
        systemic_factor = systemic_factor.sort_index()
        systemic_factor = systemic_factor.ffill().fillna(0)

    historical_df['systemic_factor'] = historical_df[date_col].map(systemic_factor.to_dict())

    # idiosyncratic_factor = norm.ppf(historical_df['idiosyncratic_score'].clip(0.001, 0.999))
    idiosyncratic_factor = norm.ppf(historical_df['idiosyncratic_score'])
    R = asset_correlation
    # asset_value = np.sqrt(R) * idiosyncratic_factor + np.sqrt(1 - R) * historical_df['systemic_factor']
    # asset value calculated as in merton model was distorting distirbutions, thus changed to:
    conditional_z = (idiosyncratic_factor + np.sqrt(R) * historical_df['systemic_factor'])/np.sqrt(1 - R)
    historical_df['simulated_score'] = norm.cdf(conditional_z)

    # --- 4.4. Map scores to ratings ---
    logger.info("Mapping simulated scores to ratings for historical sample.")
    historical_df['simulated_rating'] = _apply_score_bounds_to_ratings(
        historical_df['simulated_score'], score_to_rating_bounds
    )

    historical_df.drop(
        columns=['idiosyncratic_score', 'systemic_factor'],
        inplace=True,
    )

    # --- 5. Handle New and Existing Clients in Application Sample ---
    logger.info("Calculating migration matrix from historical simulated ratings.")
    simulated_migration_matrix = calculate_migration_matrix(
        historical_df,
        id_col=loan_id_col,
        date_col=date_col,
        rating_col='simulated_rating',
    )

    # --- 5.1. Handle "New Clients" ---
    if not new_clients_df.empty:
        logger.info("Simulating new client scores and ratings.")
        num_new_clients = new_clients_df[loan_id_col].nunique()
        n_new_bad = int(bad_proportion * num_new_clients)
        n_new_good = num_new_clients - n_new_bad
        new_client_scores_good, new_client_scores_bad = generate_calibrated_scores(bmf, gamma, n_new_good, n_new_bad)
        new_client_scores = np.concat((new_client_scores_good, new_client_scores_bad))
        new_client_score_map = dict(zip(new_clients_df[loan_id_col].unique(), new_client_scores))
        new_clients_df['simulated_score'] = new_clients_df[loan_id_col].map(new_client_score_map)
        new_clients_df['simulated_rating'] = _apply_score_bounds_to_ratings(
            new_clients_df['simulated_score'], score_to_rating_bounds
        )
        new_clients_df = _apply_migrations_vectorized(
            new_clients_df,
            simulated_migration_matrix,
            'simulated_rating',
            date_col,
            loan_id_col,
            keep_first_rating=True
        )

    # --- 5.2. Handle "Existing Clients" ---
    if not existing_clients_df.empty:
        logger.info("Simulating existing client rating migrations.")
        last_historical_ratings = historical_df.groupby(loan_id_col)['simulated_rating'].last()
        existing_clients_df['last_historical_rating'] = existing_clients_df[loan_id_col].map(last_historical_ratings)
        existing_clients_df = _apply_migrations_vectorized(
            existing_clients_df,
            simulated_migration_matrix,
            'last_historical_rating',
            date_col,
            loan_id_col,
            keep_first_rating=False)

    # --- 6. Combine and Finalize ---
    logger.info("Combining historical, new, and existing client simulations.")
    simulated_portfolio_df = pd.concat([historical_df, new_clients_df, existing_clients_df, defaulted_df], ignore_index=True)
    
    # at this point we would like to calculate long time average PD per rating for new simulated ratings and for old ratings
    # we need to do that on historical data only, as application data is not representative
    logger.info("Calculating long-term average PD from historical simulated ratings.")
    
    simulated_pd = historical_df.groupby([date_col, 'simulated_rating'])[into_default_flag_col].mean()
    simulated_pd_lra = simulated_pd.groupby('simulated_rating').mean().to_dict()
    
    observed_pd = historical_df.groupby([date_col, rating_col])[into_default_flag_col].mean()
    observed_pd_lra = observed_pd.groupby(rating_col).mean().to_dict()
    simulated_portfolio_df['simulated_pd'] = simulated_portfolio_df['simulated_rating'].map(simulated_pd_lra)
    simulated_portfolio_df['observed_pd'] = simulated_portfolio_df[rating_col].map(observed_pd_lra)
    
    default_rating = simulated_portfolio_df.loc[simulated_portfolio_df[default_col] == 1, rating_col].unique()[0]

    simulated_portfolio_df.loc[simulated_portfolio_df[default_col] == 1, ['simulated_rating','simulated_pd', 'observed_pd']] = [default_rating, 1.0, 1.0]
    elapsed = time.time() - start_time
    logger.info(f"Portfolio simulation completed in {elapsed:.2f} seconds. Final sample size: {len(simulated_portfolio_df)}")
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
    # Create a manual mapping by checking each score against the bounds
    ratings = pd.Series(index=scores.index, dtype=object)
    
    # Sort ratings by their bounds for consistent application
    sorted_ratings = sorted(
        [(rating, bounds) for rating, bounds in score_to_rating_bounds.items() if rating != 'D'],
        key=lambda x: x[1][0]  # Sort by lower bound
    )
    
    # Apply ratings one by one based on their bounds
    for score_idx, score in enumerate(scores):
        # Default to the highest rating if no match is found
        assigned_rating = sorted_ratings[-1][0]
        
        # Check each rating's bounds
        for rating, (min_bound, max_bound) in sorted_ratings:
            if min_bound <= score < max_bound:
                assigned_rating = rating
                break
            # Handle edge case for the maximum value
            elif score == max_bound and max_bound == sorted_ratings[-1][1][1]:
                assigned_rating = rating
                break
                
        ratings.iloc[score_idx] = assigned_rating
    
    return ratings

def _infer_systemic_factor(
    historical_df: pd.DataFrame, date_col: str, rating_col: str, loan_id_col: str, score_col: str
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
    # first we need to sort ratings by their mean score ascendingly
    rating_order = historical_df.groupby(rating_col)[score_col].mean().sort_values().index.tolist()
    
    historical_df['rating_cat'] = pd.Categorical(
        historical_df[rating_col], categories=rating_order, ordered=True
    )
    historical_df['rating_code'] = historical_df['rating_cat'].cat.codes

    # Calculate rating changes period over period for each loan
    historical_df['prev_rating_code'] = historical_df.groupby(loan_id_col)[
        'rating_code'
    ].shift(1)

    # Determine upgrades, downgrades, and stable ratings
    delta = (
        historical_df['rating_code'] - historical_df['prev_rating_code']
    )
    # historical_df['change'] = np.select(
    #     [delta > 0, delta < 0], ['Downgrade', 'Upgrade'], default='Stable'
    # )

    historical_df['change'] = 'Stable'
    historical_df.loc[delta < 0, 'change'] = 'Upgrade'
    historical_df.loc[delta > 0, 'change'] = 'Downgrade'

    # Calculate net upgrade percentage for each period
    migrations_by_date = (
        historical_df.groupby([date_col, 'change'])
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
    systemic_factor_values = norm.ppf(prob_of_upgrade.clip(0.001, 0.999)) # Clip for stability
    systemic_factor = pd.Series(systemic_factor_values, index=prob_of_upgrade.index, name='systemic_factor')
    return systemic_factor


def _apply_migrations_vectorized(
        df, 
        migration_matrix, 
        rating_col, 
        date_col,
        loan_id_col,
        keep_first_rating: bool = True
        ):
    """Apply migrations using vectorized operations"""
    # df = existing_clients_df
    # migration_matrix = simulated_migration_matrix
    # rating_col = 'last_historical_rating'
    # keep_first_rating=False
    # Pre-compute rating mappings
    rating_to_idx = {rating: idx for idx, rating in enumerate(migration_matrix.index)}
    idx_to_rating = {idx: rating for rating, idx in rating_to_idx.items()}
    
    # Convert ratings to indices for faster lookup
    df['rating_idx'] = df[rating_col].map(rating_to_idx)
    
    # Precompute cumulative probabilities for each rating
    cum_probs = migration_matrix.cumsum(axis=1).values

    # Prepare an array to hold new ratings
    df['new_rating'] = None

    # Mark first observation for each loan
    first_obs_mask = ~df.duplicated(subset=[loan_id_col], keep='first')

    for date, group in df.groupby(date_col):
        df.loc[df[date_col] > date, 'new_rating'] = None
        n_obs = len(group)
        if n_obs == 0:
            continue
        else:
            random_values = np.random.rand(n_obs)
            if date == df[date_col].min():
                current_ratings = group['rating_idx'].values
            else:
                current_ratings = df.loc[group.index, 'new_rating'].map(rating_to_idx).fillna(df['rating_idx']).astype(int).values
            # potential to cut
            # new_rating_indices = [np.searchsorted(cum_probs[int(idx)], random_values[i]) for i, idx in enumerate(current_ratings)]
            # we can potentially improve performance by avoiding list comprehension
            # let's assume all new_ratings are equal to current ratings
            # instead of using searchsorted for each, we can use broadcasting
            # but since current_ratings can be different, we will keep it simple for now
            new_rating_indices = np.zeros(n_obs, dtype=int)
            selected_cum_probs = cum_probs[current_ratings]
            
            mask_too_low = ~(random_values <= selected_cum_probs[np.arange(n_obs), current_ratings])
            
            mask_equal_or_higher = ~mask_too_low
            mask_correct = mask_equal_or_higher & (current_ratings == 0) # if current rating is the lowest and random value is <= cum prob it stays as is
            new_rating_indices[mask_correct] = current_ratings[mask_correct] 
            random_values_to_check = random_values[~mask_correct]
            current_ratings_to_check = current_ratings[~mask_correct]
            
            ratings_checked = [np.searchsorted(cum_probs[int(idx)], random_values_to_check[i]) for i, idx in enumerate(current_ratings_to_check)]
            new_rating_indices[~mask_correct] = ratings_checked

            
            # this indicates that current carting is equal or higher than new rating
            # we can now apply searchsorted for these rows where above check fails.
            

            new_ratings = [idx_to_rating[new_rating_idx] for new_rating_idx in new_rating_indices]
            df.loc[group.index, 'new_rating'] = new_ratings
            if keep_first_rating:
                group_first_obs_mask = list(set(group.index).intersection(set(df[first_obs_mask].index)))
                df.loc[group_first_obs_mask, 'new_rating'] = df.loc[group_first_obs_mask, rating_col]
            df['new_rating'] = df.groupby(loan_id_col)['new_rating'].ffill()

    df['simulated_rating'] = df['new_rating']
    df.drop(columns=['rating_idx', 'new_rating'], inplace=True)
    return df

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
