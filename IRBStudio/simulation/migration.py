"""
This module provides tools for calculating rating migration matrices from historical data.
"""
import pandas as pd
import numpy as np
from typing import List

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

    # Ensure date column is of datetime type and sort
    df = data[[id_col, date_col, rating_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=[id_col, date_col]).dropna()

    # Get all unique ratings from the original data to build a complete matrix
    # Convert to string to handle mixed types (e.g., int ratings and 'D' for default)
    all_ratings = sorted([str(r) for r in data[rating_col].dropna().unique()])
    
    if len(df) <= 1 or len(df[id_col].unique()) <= 1:
        # If there are no transitions possible, return an identity matrix
        identity_matrix = pd.DataFrame(np.identity(len(all_ratings)), 
                                      index=all_ratings, 
                                      columns=all_ratings)
        return identity_matrix

    # Create transition pairs for each loan
    result_rows = []
    for loan_id, group in df.groupby(id_col):
        if len(group) <= 1:
            continue
            
        sorted_group = group.sort_values(by=date_col)
        from_ratings = sorted_group[rating_col].iloc[:-1].astype(str).tolist()
        to_ratings = sorted_group[rating_col].iloc[1:].astype(str).tolist()
        
        for from_rating, to_rating in zip(from_ratings, to_ratings):
            result_rows.append({'from': from_rating, 'to': to_rating})
    
    if not result_rows:
        # If there are no transitions, return an identity matrix
        identity_matrix = pd.DataFrame(np.identity(len(all_ratings)), 
                                      index=all_ratings, 
                                      columns=all_ratings)
        return identity_matrix
    
    # Convert to DataFrame and calculate transition matrix
    transitions_df = pd.DataFrame(result_rows)
    
    # Create a crosstab and calculate probabilities
    counts = pd.crosstab(transitions_df['from'], transitions_df['to'])
    
    # Reindex to ensure all ratings are present
    counts = counts.reindex(index=all_ratings, columns=all_ratings, fill_value=0)
    
    # Calculate probabilities - rows should sum to 1
    probabilities = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    
    # For any row with all zeros (no transitions observed), set diagonal to 1
    for rating in all_ratings:
        if probabilities.loc[rating].sum() == 0:
            probabilities.loc[rating, rating] = 1.0
    
    # Reset index and column names
    probabilities.index.name = None
    probabilities.columns.name = None
    
    return probabilities
