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
    # data = historical_df
    # id_col=loan_id_col
    # rating_col='simulated_rating'
    # date_col = 'reporting_date'

    if not all(col in data.columns for col in [id_col, date_col, rating_col]):
        raise ValueError("One or more specified columns are not in the DataFrame.")
    data['prev_rating'] = data.groupby(id_col)[rating_col].shift(1)
    transitions = data.loc[data['prev_rating'].notna()]
    migration_counts = transitions.groupby('prev_rating')[rating_col].value_counts(normalize=False).unstack().fillna(0)
    
    # migration_counts = pd.crosstab(transitions['prev_rating'], transitions[rating_col])
    migration_matrix = migration_counts.div(migration_counts.sum(axis=1), axis=0).fillna(0)
    migration_matrix.index.name = None
    migration_matrix.columns.name = None
    return migration_matrix
