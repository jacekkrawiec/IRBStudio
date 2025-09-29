"""
This module provides tools for calculating rating migration matrices from historical data.
"""
import pandas as pd
import numpy as np
from typing import List

from irbstudio import data

def calculate_migration_matrix(
    data: pd.DataFrame,
    id_col: str,
    date_col: str,
    rating_col: str
) -> pd.DataFrame:
    """
    Calculates the rating migration matrix from historical loan data.

    The migration matrix shows the probability of a loan transitioning from one
    rating to another over a single period.

    Args:
        data (pd.DataFrame): DataFrame containing loan-level data with at least
                             an ID, a date, and a rating column.
        id_col (str): The name of the column containing the unique loan identifier.
        date_col (str): The name of the column containing the observation date.
        rating_col (str): The name of the column containing the rating.

    Returns:
        pd.DataFrame: A square DataFrame where both the index and columns are the
                      unique ratings. Each cell (i, j) contains the probability
                      of migrating from rating i to rating j.
    """
    if not all(col in data.columns for col in [id_col, date_col, rating_col]):
        raise ValueError("One or more specified columns are not in the DataFrame.")

    # Ensure ratings are categorical for efficient mapping
    ratings = pd.Categorical(data[rating_col])
    
    # Sort by id and date to ensure correct order
    rating_codes = ratings.codes
    loan_ids = data[id_col].values

    # Find previous rating for each row (within each loan)
    prev_rating_codes = np.empty_like(rating_codes)
    prev_rating_codes[0] = -1  # First row has no previous
    prev_rating_codes[1:] = np.where(
        loan_ids[1:] == loan_ids[:-1],
        rating_codes[:-1],
        -1
    )

    # Only consider rows with a previous rating
    mask = prev_rating_codes != -1
    from_codes = prev_rating_codes[mask]
    to_codes = rating_codes[mask]

    # Build 2D histogram (transition counts)
    n_ratings = len(ratings.categories)
    migration_counts = np.zeros((n_ratings, n_ratings), dtype=np.int64)
    np.add.at(migration_counts, (from_codes, to_codes), 1)

    # Normalize rows to get probabilities
    row_sums = migration_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        migration_matrix = np.divide(migration_counts, row_sums, where=row_sums!=0)
        migration_matrix = np.nan_to_num(migration_matrix)

    # Convert to DataFrame with proper labels
    migration_df = pd.DataFrame(
        migration_matrix,
        index=ratings.categories,
        columns=ratings.categories
    )

    return migration_df

