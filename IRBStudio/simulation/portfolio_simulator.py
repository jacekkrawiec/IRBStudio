import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Union, Tuple
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


class PortfolioSimulator:
    """
    Simulates portfolio performance based on historical data and target parameters.
    
    This class provides an object-oriented implementation of the portfolio simulation
    logic, separating the deterministic preparation steps from the stochastic simulation
    steps. This design is optimized for:
    
    1. Memory efficiency in Monte Carlo simulations
    2. Clean separation of preparation and simulation logic
    3. Ability to reuse fitted models across multiple simulations
    4. Extensibility for new simulation features
    
    The simulation follows a procedurally-faithful approach by:
    1. Segmenting the portfolio into historical and application samples
    2. Fitting models to historical data
    3. Simulating rating migrations for existing clients
    4. Drawing scores for new clients from a fitted distribution
    5. Combining the segments and applying the final calibration
    """
    
    def __init__(
        self,
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
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the portfolio simulator with dataset and configuration parameters.
        
        Args:
            portfolio_df: The full portfolio dataset, including history
            score_to_rating_bounds: Dict mapping rating to (min_score, max_score)
            rating_col: The name of the column containing rating grades
            loan_id_col: The name of the column containing unique loan identifiers
            date_col: The name of the column containing the snapshot date
            default_col: The name of the column containing the default flag
            into_default_flag_col: The name of the column flagging the period of default entry
            score_col: The name of the column containing the current model's score
            application_start_date: The date from which the application sample begins
                If not provided, it defaults to a 12-month window ending on the most recent date
            asset_correlation: The asset correlation parameter (R) used in the Merton model
                Defaults to 0.15, a common value for mortgages
            exposure_col: The name of the exposure column
                If provided, it will be carried through the simulation
            target_auc: The target AUC for new client scores
                If provided, new client scores will be calibrated to achieve this AUC
            random_seed: Optional random seed for reproducibility
        """
        # Store configuration parameters
        self.score_to_rating_bounds = score_to_rating_bounds
        self.rating_col = rating_col
        self.loan_id_col = loan_id_col
        self.date_col = date_col
        self.default_col = default_col
        self.into_default_flag_col = into_default_flag_col
        self.score_col = score_col
        self.application_start_date = application_start_date
        self.asset_correlation = asset_correlation
        self.exposure_col = exposure_col
        self.target_auc = target_auc
        
        # Set random seed if provided
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize data containers
        self.portfolio_df = portfolio_df.copy()
        self.portfolio_df[self.date_col] = pd.to_datetime(self.portfolio_df[self.date_col])
        self.portfolio_df = self.portfolio_df.sort_values(by=[self.loan_id_col, self.date_col])
        
        # Status flags
        self.is_prepared = False
        
        # Placeholder for simulation components (will be populated during preparation)
        self.defaulted_df = None
        self.historical_df = None
        self.application_df = None
        self.existing_clients_df = None
        self.new_clients_df = None
        self.bad_proportion = None
        self.simulated_migration_matrix = None
        self.beta_mixture = None
        self.gamma = None
        self.default_rating = None
        self.systemic_factor = None
        self.simulated_pd_lra = None
        self.observed_pd_lra = None
        
        # Initialize logger
        self.logger = get_logger(__name__)
    
    def prepare_simulation(self) -> 'PortfolioSimulator':
        """
        Prepare simulation by performing all deterministic steps.
        
        This method:
        1. Segments the portfolio data
        2. Fits the Beta Mixture Model to historical scores
        3. Calculates migration matrices
        4. Infers systemic factors
        
        Returns:
            self: Returns self for method chaining
        """
        # Set random seed for reproducibility during preparation
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        start_time = time.time()
        self.logger.info("Starting portfolio simulation preparation.")
        
        # Segment the portfolio
        self._segment_portfolio()
        
        # Fit Beta Mixture Model and prepare calibration factors
        self._fit_beta_mixture()
        
        # Infer systemic factor and generate historical simulated ratings
        self._simulate_historical_ratings()
        
        # Calculate migration matrix from historical ratings
        self._calculate_migration_matrix()
        
        # Calculate long-term average PDs for observed and simulated ratings
        self._calculate_long_term_pd()
        
        # Mark preparation as complete
        self.is_prepared = True
        
        elapsed = time.time() - start_time
        self.logger.info(f"Portfolio simulation preparation completed in {elapsed:.2f} seconds.")
        
        return self
    
    def simulate_once(self, random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Run a single iteration of the simulation.

        Args:
            random_seed: Optional random seed for reproducibility
            
        Returns:
            pd.DataFrame: Simulated portfolio with new ratings and PDs
        """
        if not self.is_prepared:
            self.prepare_simulation()
        
        # Set the random seed for this simulation
        if random_seed is not None:
            np.random.seed(random_seed)
        elif self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        start_time = time.time()
        self.logger.info("Starting portfolio simulation iteration.")        # Simulate new clients
        simulated_new_clients_df = self._simulate_new_clients() if not self.new_clients_df.empty else pd.DataFrame()
        
        # Simulate existing clients
        simulated_existing_clients_df = self._simulate_existing_clients() if not self.existing_clients_df.empty else pd.DataFrame()
        
        # Combine all segments
        simulated_portfolio_df = pd.concat(
            [self.historical_df, simulated_new_clients_df, simulated_existing_clients_df, self.defaulted_df],
            ignore_index=True
        )
        
        # Apply PD mapping
        simulated_portfolio_df['simulated_pd'] = simulated_portfolio_df['simulated_rating'].map(self.simulated_pd_lra)
        simulated_portfolio_df['observed_pd'] = simulated_portfolio_df[self.rating_col].map(self.observed_pd_lra)
        
        # Handle defaulted loans
        simulated_portfolio_df.loc[
            simulated_portfolio_df[self.default_col] == 1, 
            ['simulated_rating', 'simulated_pd', 'observed_pd']
        ] = [self.default_rating, 1.0, 1.0]
        
        elapsed = time.time() - start_time
        self.logger.info(f"Portfolio simulation iteration completed in {elapsed:.2f} seconds. Final sample size: {len(simulated_portfolio_df)}")
        
        return simulated_portfolio_df
    
    def run_monte_carlo(self, num_iterations: int, random_seed: Optional[int] = None) -> List[pd.DataFrame]:
        """
        Run multiple simulations for Monte Carlo analysis.
        
        Args:
            num_iterations: Number of simulations to run
            random_seed: Optional base random seed for reproducibility
            
        Returns:
            List[pd.DataFrame]: List of simulated portfolios, one for each iteration
        """
        if not self.is_prepared:
            self.prepare_simulation()
        
        results = []
        
        self.logger.info(f"Starting Monte Carlo simulation with {num_iterations} iterations.")
        start_time = time.time()
        
        for i in range(num_iterations):
            # Set iteration-specific seed if base seed was provided
            iter_seed = None
            if random_seed is not None:
                iter_seed = random_seed + i
            
            self.logger.info(f"Running Monte Carlo iteration {i+1}/{num_iterations}")
            simulated_df = self.simulate_once(random_seed=iter_seed)
            results.append(simulated_df)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Monte Carlo simulation completed in {elapsed:.2f} seconds. Average time per iteration: {elapsed/num_iterations:.2f} seconds.")
        
        return results
    
    def _segment_portfolio(self):
        """Segment the portfolio into historical and application samples."""
        self.logger.info("Segmenting portfolio into historical and application samples.")
        
        # Determine application start date if not provided
        if self.application_start_date is None:
            most_recent_date = self.portfolio_df[self.date_col].max()
            self.application_start_date = most_recent_date - DateOffset(months=11)
            self.application_start_date = self.application_start_date.replace(day=1)
            self.logger.info(f"No application_start_date provided. Using default: {self.application_start_date}")
        
        # Separate defaulted loans
        self.defaulted_df = self.portfolio_df[self.portfolio_df[self.default_col] == 1].copy()
        clean_portfolio_df = self.portfolio_df.loc[~self.portfolio_df.index.isin(self.defaulted_df.index)].copy()
        
        # Split into historical and application periods
        self.historical_df = clean_portfolio_df[clean_portfolio_df[self.date_col] < self.application_start_date].copy()
        self.application_df = clean_portfolio_df[clean_portfolio_df[self.date_col] >= self.application_start_date].copy()
        
        self.logger.info(f"Historical sample size: {len(self.historical_df)}; Application sample size: {len(self.application_df)}")
        
        # Validate segments
        if self.historical_df.empty:
            self.logger.error("Historical data is empty. Cannot proceed with simulation.")
            raise ValueError("Historical data is empty. Cannot proceed with simulation.")
        if self.application_df.empty:
            self.logger.error(f"No application data found from date {self.application_start_date} onwards. Cannot proceed with simulation.")
            raise ValueError(
                f"No application data found from date {self.application_start_date} onwards. "
                "Cannot proceed with simulation."
            )
        
        # Further segment application sample into new and existing clients
        self.logger.info("Segmenting application sample into new and existing clients.")
        historical_ids = set(self.historical_df[self.loan_id_col].unique())
        application_ids = set(self.application_df[self.loan_id_col].unique())
        
        existing_client_ids = historical_ids.intersection(application_ids)
        new_client_ids = application_ids - historical_ids
        
        self.existing_clients_df = self.application_df[
            self.application_df[self.loan_id_col].isin(existing_client_ids)
        ].copy()
        self.new_clients_df = self.application_df[
            self.application_df[self.loan_id_col].isin(new_client_ids)
        ].copy()
        
        self.logger.info(f"Existing clients: {len(self.existing_clients_df)}, New clients: {len(self.new_clients_df)}")
        
        # Calculate default rates for later use
        self.defaulted_facility_ids = set(
            self.historical_df.loc[self.historical_df[self.into_default_flag_col] == 1, self.loan_id_col].unique()
        )
        self.non_defaulted_facility_ids = set(
            self.historical_df.loc[self.historical_df[self.into_default_flag_col] == 0, self.loan_id_col].unique()
        )
        
        self.num_defaulted_facilities = len(self.defaulted_facility_ids)
        self.num_non_defaulted_facilities = len(self.non_defaulted_facility_ids)
        self.bad_proportion = self.num_defaulted_facilities / self.historical_df[self.loan_id_col].nunique()
        
        # Get default rating for later use
        if not self.defaulted_df.empty:
            self.default_rating = self.defaulted_df.loc[self.defaulted_df[self.default_col] == 1, self.rating_col].unique()[0]
        else:
            self.default_rating = 'D'  # Default fallback if no defaults in the dataset
    
    def _fit_beta_mixture(self):
        """Fit Beta Mixture Model to historical scores."""
        self.logger.info("Fitting Beta Mixture Model to historical scores.")
        self.beta_mixture = BetaMixtureFitter(n_components=2)
        
        fit_df = self.historical_df[[self.score_col, self.into_default_flag_col]].dropna()
        X_fit = fit_df[self.score_col].values
        y_fit = fit_df[self.into_default_flag_col].values
        
        try:
            self.beta_mixture.fit(X_fit, y_fit)
            self.logger.info("Supervised fitting of Beta Mixture Model succeeded.")
        except Exception as e:
            self.logger.warning(f"Supervised fitting failed: {e}. Falling back to unsupervised.")
            non_default_scores = self.historical_df.loc[self.historical_df[self.default_col] == 0, self.score_col].dropna()
            if len(non_default_scores) < 10:
                synthetic_scores = np.random.beta(2, 5, size=100)
                non_default_scores = pd.Series(synthetic_scores)
            clipped_scores = non_default_scores.clip(0.001, 0.999).values
            self.beta_mixture.fit(clipped_scores)
            self.logger.info("Unsupervised fitting of Beta Mixture Model succeeded.")
        
        # Calculate calibration factor if target AUC is provided
        if self.target_auc is not None and 0.5 <= self.target_auc < 1.0:
            self.gamma = find_auc_calibration_factor(self.beta_mixture, self.target_auc)
        else:
            # Default gamma of 2.0 provides a reasonable separation when no target is specified
            self.gamma = 2.0
            self.logger.info(f"No valid target AUC provided. Using default gamma of {self.gamma}.")
    
    def _simulate_historical_ratings(self):
        """Generate simulated ratings for historical data."""
        # Generate idiosyncratic scores
        self.logger.info("Generating idiosyncratic scores for historical sample.")
        scores_good, scores_bad = generate_calibrated_scores(
            self.beta_mixture, 
            self.gamma, 
            self.num_non_defaulted_facilities, 
            self.num_defaulted_facilities
        )
        
        nd_scores_dict = dict(zip(self.non_defaulted_facility_ids, scores_good))
        d_scores_dict = dict(zip(self.defaulted_facility_ids, scores_bad))
        
        self.historical_df['idiosyncratic_score'] = np.nan
        mask_default = self.historical_df[self.into_default_flag_col] == 1
        mask_non_default = self.historical_df[self.into_default_flag_col] == 0
        
        self.historical_df.loc[mask_default, 'idiosyncratic_score'] = (
            self.historical_df.loc[mask_default, self.loan_id_col].map(d_scores_dict)
        )
        self.historical_df.loc[mask_non_default, 'idiosyncratic_score'] = (
            self.historical_df.loc[mask_non_default, self.loan_id_col].map(nd_scores_dict)
        )
        
        # Infer systemic factor
        self.logger.info("Inferring systemic risk factor from historical migrations.")
        self.systemic_factor = self._infer_systemic_factor()
        
        # Handle missing dates in systemic factor
        if len(self.systemic_factor) < len(self.historical_df[self.date_col].unique()):
            missing_dates = set(self.historical_df[self.date_col].unique()) - set(self.systemic_factor.index)
            for missing_date in missing_dates:  
                self.systemic_factor.loc[missing_date] = np.nan
            self.systemic_factor = self.systemic_factor.sort_index()
            self.systemic_factor = self.systemic_factor.ffill().fillna(0)
        
        # Map systemic factor to historical data
        self.historical_df['systemic_factor'] = self.historical_df[self.date_col].map(self.systemic_factor.to_dict())
        
        # Calculate simulated scores using Merton model
        idiosyncratic_factor = norm.ppf(self.historical_df['idiosyncratic_score'])
        R = self.asset_correlation
        conditional_z = (idiosyncratic_factor + np.sqrt(R) * self.historical_df['systemic_factor']) / np.sqrt(1 - R)
        self.historical_df['simulated_score'] = norm.cdf(conditional_z)
        
        # Map scores to ratings
        self.logger.info("Mapping simulated scores to ratings for historical sample.")
        self.historical_df['simulated_rating'] = self._apply_score_bounds_to_ratings(
            self.historical_df['simulated_score']
        )
        
        # Clean up temporary columns
        self.historical_df.drop(columns=['idiosyncratic_score', 'systemic_factor'], inplace=True)
    
    def _calculate_migration_matrix(self):
        """Calculate migration matrix from historical simulated ratings."""
        self.logger.info("Calculating migration matrix from historical simulated ratings.")
        self.simulated_migration_matrix = calculate_migration_matrix(
            self.historical_df,
            id_col=self.loan_id_col,
            date_col=self.date_col,
            rating_col='simulated_rating'
        )
    
    def _calculate_long_term_pd(self):
        """Calculate long-term average PDs for observed and simulated ratings."""
        self.logger.info("Calculating long-term average PD from historical simulated ratings.")
        
        # Calculate simulated PDs
        simulated_pd = self.historical_df.groupby([self.date_col, 'simulated_rating'])[self.into_default_flag_col].mean()
        self.simulated_pd_lra = simulated_pd.groupby('simulated_rating').mean().to_dict()
        
        # Calculate observed PDs
        observed_pd = self.historical_df.groupby([self.date_col, self.rating_col])[self.into_default_flag_col].mean()
        self.observed_pd_lra = observed_pd.groupby(self.rating_col).mean().to_dict()
        
        # Ensure default rating has PD of 1.0
        self.simulated_pd_lra[self.default_rating] = 1.0
        self.observed_pd_lra[self.default_rating] = 1.0
    
    def _simulate_new_clients(self) -> pd.DataFrame:
        """Simulate ratings for new clients."""
        if self.new_clients_df.empty:
            return pd.DataFrame()
        
        self.logger.info("Simulating new client scores and ratings.")
        
        # Copy the dataframe to avoid modifying the original
        new_clients_df = self.new_clients_df.copy()
        
        # Generate calibrated scores for new clients
        num_new_clients = new_clients_df[self.loan_id_col].nunique()
        n_new_bad = int(self.bad_proportion * num_new_clients)
        n_new_good = num_new_clients - n_new_bad
        
        new_client_scores_good, new_client_scores_bad = generate_calibrated_scores(
            self.beta_mixture, 
            self.gamma, 
            n_new_good, 
            n_new_bad
        )
        
        # Combine scores and map to loans
        new_client_scores = np.concatenate((new_client_scores_good, new_client_scores_bad))
        new_client_score_map = dict(zip(new_clients_df[self.loan_id_col].unique(), new_client_scores))
        
        # Apply scores and map to ratings
        new_clients_df['simulated_score'] = new_clients_df[self.loan_id_col].map(new_client_score_map)
        new_clients_df['simulated_rating'] = self._apply_score_bounds_to_ratings(new_clients_df['simulated_score'])
        
        # Apply migrations
        new_clients_df = self._apply_migrations_vectorized(
            new_clients_df,
            self.simulated_migration_matrix,
            'simulated_rating',
            keep_first_rating=True
        )
        
        return new_clients_df
    
    def _simulate_existing_clients(self) -> pd.DataFrame:
        """Simulate ratings for existing clients."""
        if self.existing_clients_df.empty:
            return pd.DataFrame()
        
        self.logger.info("Simulating existing client rating migrations.")
        
        # Copy the dataframe to avoid modifying the original
        existing_clients_df = self.existing_clients_df.copy()
        
        # Get the last historical rating for each existing client
        last_historical_ratings = self.historical_df.groupby(self.loan_id_col)['simulated_rating'].last()
        existing_clients_df['last_historical_rating'] = existing_clients_df[self.loan_id_col].map(last_historical_ratings)
        
        # Apply migrations
        existing_clients_df = self._apply_migrations_vectorized(
            existing_clients_df,
            self.simulated_migration_matrix,
            'last_historical_rating',
            keep_first_rating=False
        )
        
        return existing_clients_df
    
    def _apply_score_bounds_to_ratings(self, scores: pd.Series) -> pd.Series:
        """
        Map scores to ratings using explicit user-provided score bounds for each rating.

        Args:
            scores: pd.Series of scores to map

        Returns:
            pd.Series of ratings
        """
        # Create a manual mapping by checking each score against the bounds
        ratings = pd.Series(index=scores.index, dtype=object)
        
        # Sort ratings by their bounds for consistent application
        sorted_ratings = sorted(
            [(rating, bounds) for rating, bounds in self.score_to_rating_bounds.items() if rating != self.default_rating],
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
    
    def _infer_systemic_factor(self) -> pd.Series:
        """
        Infers a historical systemic risk factor (M_t) from observed rating migrations.

        This function calculates the net upgrade/downgrade percentage for each period
        and uses the inverse CDF of a standard normal distribution to transform this
        into a systemic factor.

        Returns:
            pd.Series: A series of systemic factors indexed by date.
        """
        # First we need to sort ratings by their mean score ascendingly
        rating_order = self.historical_df.groupby(self.rating_col)[self.score_col].mean().sort_values().index.tolist()
        
        # Create categorical ratings for easier comparison
        self.historical_df['rating_cat'] = pd.Categorical(
            self.historical_df[self.rating_col], categories=rating_order, ordered=True
        )
        self.historical_df['rating_code'] = self.historical_df['rating_cat'].cat.codes

        # Calculate rating changes period over period for each loan
        self.historical_df['prev_rating_code'] = self.historical_df.groupby(self.loan_id_col)[
            'rating_code'
        ].shift(1)

        # Determine upgrades, downgrades, and stable ratings
        delta = self.historical_df['rating_code'] - self.historical_df['prev_rating_code']
        
        self.historical_df['change'] = 'Stable'
        self.historical_df.loc[delta < 0, 'change'] = 'Upgrade'
        self.historical_df.loc[delta > 0, 'change'] = 'Downgrade'

        # Calculate net upgrade percentage for each period
        migrations_by_date = (
            self.historical_df.groupby([self.date_col, 'change'])
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
        systemic_factor_values = norm.ppf(prob_of_upgrade.clip(0.001, 0.999))  # Clip for stability
        systemic_factor = pd.Series(systemic_factor_values, index=prob_of_upgrade.index, name='systemic_factor')
        
        return systemic_factor
    
    def _apply_migrations_vectorized(
            self, 
            df: pd.DataFrame,
            migration_matrix: pd.DataFrame,
            rating_col: str,
            keep_first_rating: bool = True
            ) -> pd.DataFrame:
        """
        Apply migrations using vectorized operations.
        
        Args:
            df: DataFrame containing the loans to migrate
            migration_matrix: Matrix of migration probabilities
            rating_col: Column containing the starting rating
            keep_first_rating: Whether to keep the first rating or migrate it
            
        Returns:
            DataFrame with simulated ratings
        """
        # Create a copy to avoid modifying the input
        df = df.copy()
        
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
        first_obs_mask = ~df.duplicated(subset=[self.loan_id_col], keep='first')

        for date, group in df.groupby(self.date_col):
            df.loc[df[self.date_col] > date, 'new_rating'] = None
            n_obs = len(group)
            if n_obs == 0:
                continue
            else:
                random_values = np.random.rand(n_obs)
                if date == df[self.date_col].min():
                    current_ratings = group['rating_idx'].values
                else:
                    current_ratings = df.loc[group.index, 'new_rating'].map(rating_to_idx).fillna(df['rating_idx']).astype(int).values
                
                # Initialize new ratings array
                new_rating_indices = np.zeros(n_obs, dtype=int)
                selected_cum_probs = cum_probs[current_ratings]
                
                # Identify ratings that need to be migrated
                mask_too_low = ~(random_values <= selected_cum_probs[np.arange(n_obs), current_ratings])
                mask_equal_or_higher = ~mask_too_low
                mask_correct = mask_equal_or_higher & (current_ratings == 0)  # Lowest rating and random value <= cum prob
                
                # Keep correct ratings unchanged
                new_rating_indices[mask_correct] = current_ratings[mask_correct] 
                
                # Process ratings that need migration
                random_values_to_check = random_values[~mask_correct]
                current_ratings_to_check = current_ratings[~mask_correct]
                
                # Find new ratings based on migration probabilities
                ratings_checked = [np.searchsorted(cum_probs[int(idx)], random_values_to_check[i]) 
                                for i, idx in enumerate(current_ratings_to_check)]
                new_rating_indices[~mask_correct] = ratings_checked

                # Convert indices back to ratings
                new_ratings = [idx_to_rating[new_rating_idx] for new_rating_idx in new_rating_indices]
                df.loc[group.index, 'new_rating'] = new_ratings
                
                # Handle first observation if needed
                if keep_first_rating:
                    group_first_obs_mask = list(set(group.index).intersection(set(df[first_obs_mask].index)))
                    df.loc[group_first_obs_mask, 'new_rating'] = df.loc[group_first_obs_mask, rating_col]
                
                # Forward fill ratings within each loan
                df['new_rating'] = df.groupby(self.loan_id_col)['new_rating'].ffill()

        # Set final simulated rating and clean up
        df['simulated_rating'] = df['new_rating']
        df.drop(columns=['rating_idx', 'new_rating'], inplace=True)
        
        return df


# Backward compatibility function
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
    # Create simulator instance
    simulator = PortfolioSimulator(
        portfolio_df=portfolio_df,
        score_to_rating_bounds=score_to_rating_bounds,
        rating_col=rating_col,
        loan_id_col=loan_id_col,
        date_col=date_col,
        default_col=default_col,
        into_default_flag_col=into_default_flag_col,
        score_col=score_col,
        application_start_date=application_start_date,
        asset_correlation=asset_correlation,
        exposure_col=exposure_col,
        target_auc=target_auc,
    )
    
    # Run simulation
    return simulator.prepare_simulation().simulate_once()