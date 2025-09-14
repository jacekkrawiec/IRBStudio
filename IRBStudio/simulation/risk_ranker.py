"""
This module contains the Risk-Ranker, a component responsible for training
a model on historical data to rank the credit risk of new applications.
"""
from typing import List
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ..utils.logging import get_logger

logger = get_logger(__name__)

def train_risk_ranker(
    historical_data: pd.DataFrame, 
    features: List[str], 
    target: str
) -> LogisticRegression:
    """
    Trains a Logistic Regression model on historical data.

    This model's purpose is not to predict PD accurately, but to provide a stable,
    data-driven ranking of risk for the application portfolio.

    Args:
        historical_data: DataFrame containing the historical loans with known outcomes.
        features: A list of column names to be used as features for the model.
        target: The name of the target column (e.g., 'default_flag').

    Returns:
        A trained scikit-learn LogisticRegression model.
        
    Raises:
        ValueError: If the historical data is empty, or if features/target are not present.
    """
    logger.info(f"Training Risk-Ranker model with features {features} and target '{target}'.")

    if historical_data.empty:
        raise ValueError("Historical data cannot be empty for training the Risk-Ranker.")

    missing_cols = [col for col in features + [target] if col not in historical_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in historical_data: {missing_cols}")

    # Ensure target variable is binary and has both classes for robust training
    if not (historical_data[target].nunique() == 2 and historical_data[target].min() == 0 and historical_data[target].max() == 1):
        logger.warning(f"Target variable '{target}' is not binary [0, 1]. Model training may be unstable or fail.")

    X = historical_data[features]
    y = historical_data[target]

    # Simple model, no need for complex hyperparameter tuning for a ranker
    model = LogisticRegression(random_state=42, class_weight='balanced')
    
    logger.info("Fitting LogisticRegression model...")
    model.fit(X, y)
    logger.info("Risk-Ranker model training complete.")

    return model
