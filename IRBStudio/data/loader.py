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

    # Build mapping dict, skipping None values and nested dicts
    mapping_dump = mapping.model_dump()
    rename_dict = {
        v: k 
        for k, v in mapping_dump.items() 
        if v is not None and isinstance(v, str)
    }
    
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


def partition_data(
    df: pd.DataFrame, mapping: ColumnMapping
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions the portfolio into historical and application datasets.

    Uses the 'loan_type' column and the 'loan_type_mapping' from the config
    to distinguish between loan types.

    Args:
        df: The full portfolio DataFrame with canonical column names.
        mapping: The ColumnMapping object containing the loan type map.

    Returns:
        A tuple containing two DataFrames: (historical_data, application_data).

    Raises:
        ValueError: If the required columns or mappings for partitioning are missing.
    """
    logger.info("Partitioning data into historical and application sets.")
    
    # --- Validation specific to this function ---
    if not mapping.loan_type:
        raise ValueError("The 'loan_type' field in 'column_mapping' must be specified for partitioning.")
    
    required_col = mapping.loan_type
    if required_col not in df.columns:
        # This case should ideally be caught by validate_portfolio if loan_type is made required there
        # but we check again for robustness.
        raise ValueError(f"The column '{required_col}' (mapped to 'loan_type') was not found in the DataFrame.")

    type_map = mapping.loan_type_mapping
    if 'historical' not in type_map or 'application' not in type_map:
        raise ValueError(
            "The 'loan_type_mapping' in your config must contain keys for both 'historical' and 'application'."
        )
    # --- End Validation ---

    historical_identifier = type_map['historical']
    application_identifier = type_map['application']

    logger.debug(f"Identifying historical loans with value '{historical_identifier}' in column '{required_col}'")
    logger.debug(f"Identifying application loans with value '{application_identifier}' in column '{required_col}'")

    historical_df = df[df[required_col] == historical_identifier].copy()
    application_df = df[df[required_col] == application_identifier].copy()

    logger.info(f"Found {len(historical_df)} historical loans.")
    logger.info(f"Found {len(application_df)} application loans.")

    if len(historical_df) == 0:
        logger.warning("No historical loans found after partitioning. The 'Risk-Ranker' model cannot be trained.")
    if len(application_df) == 0:
        logger.warning("No application loans found after partitioning. The simulation will have no portfolio to score.")

    return historical_df, application_df

