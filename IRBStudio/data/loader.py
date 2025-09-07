"""Handles data loading, validation, and mapping."""

import yaml
import pandas as pd
from ..config.schema import Config, ColumnMapping
from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_portfolio(path: str, mapping: ColumnMapping) -> pd.DataFrame:
    """
    Loads a portfolio file (CSV or Parquet) and applies column mapping.

    Args:
        path: Path to the portfolio file (CSV or Parquet).
        mapping: ColumnMapping object from the validated config.

    Returns:
        pd.DataFrame with standardized column names.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    logger.info(f"Loading portfolio from: {path}")
    if path.lower().endswith(".csv"):
        logger.debug("Reading as CSV file.")
        df = pd.read_csv(path)
    elif path.lower().endswith((".parquet", ".pq")):
        logger.debug("Reading as Parquet file.")
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file type. Only CSV and Parquet are supported.")
    
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    original_columns = df.columns.tolist()

    # Build mapping dict, skipping None values (Pydantic v2: use model_dump())
    mapping_dict = {k: v for k, v in mapping.model_dump().items() if v is not None}
    # Invert mapping: user_col -> canonical_col
    rename_dict = {v: k for k, v in mapping_dict.items()}
    
    logger.info(f"Applying column mappings: {rename_dict}")
    df = df.rename(columns=rename_dict)
    
    # Validate required canonical columns
    validate_portfolio(df, original_columns)
    logger.info("Portfolio validation successful.")
    return df


def validate_portfolio(df: pd.DataFrame, original_columns: list):
    """
    Checks that all required canonical columns are present in the DataFrame.
    Raises ValueError with a detailed message if any are missing.
    """
    required_fields = ColumnMapping.get_required_fields()
    missing = [col for col in required_fields if col not in df.columns]
    if missing:
        error_msg = (
            f"Portfolio is missing required columns after mapping: {missing}. "
            f"Please check the 'column_mapping' section in your config file. "
            f"Original columns found in the file: {original_columns}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def load_config(path: str) -> Config:
    """Loads a YAML config file from the given path and validates it.

    Args:
        path: The file path to the YAML configuration file.

    Returns:
        A validated Config object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the YAML content does not match the Config schema.
    """
    logger.info(f"Loading configuration from: {path}")
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Pydantic does the heavy lifting of validation here
    validated_config = Config(**raw_config)
    logger.info("Configuration successfully loaded and validated.")

    return validated_config

