import pytest
import pandas as pd
from pydantic import ValidationError
from irbstudio.data.loader import partition_data
from irbstudio.config.schema import ColumnMapping

@pytest.fixture
def sample_df():
    """DataFrame with mixed historical and application loans."""
    data = {
        "loan_id": [1, 2, 3, 4, 5],
        "loan_type": ["H", "A", "H", "A", "H"],
        "default_flag": [1, 0, 0, 0, 1],
    }
    return pd.DataFrame(data)

@pytest.fixture
def valid_mapping():
    """A valid ColumnMapping with loan_type_mapping."""
    return ColumnMapping(
        loan_id="loan_id",
        loan_type="loan_type",
        default_flag="default_flag",
        exposure="exposure", # Added required field
        loan_type_mapping={"historical": "H", "application": "A"},
    )

def test_partition_data_success(sample_df, valid_mapping):
    """Test successful partitioning of data."""
    historical_df, application_df = partition_data(sample_df, valid_mapping)

    assert len(historical_df) == 3
    assert len(application_df) == 2
    assert historical_df["loan_type"].unique() == ["H"]
    assert application_df["loan_type"].unique() == ["A"]
    assert list(historical_df['loan_id']) == [1, 3, 5]
    assert list(application_df['loan_id']) == [2, 4]

def test_partition_data_no_historical_loans(sample_df, valid_mapping):
    """Test partitioning when only application loans are present."""
    app_only_df = sample_df[sample_df["loan_type"] == "A"].copy()
    historical_df, application_df = partition_data(app_only_df, valid_mapping)

    assert len(historical_df) == 0
    assert len(application_df) == 2

def test_partition_data_no_application_loans(sample_df, valid_mapping):
    """Test partitioning when only historical loans are present."""
    hist_only_df = sample_df[sample_df["loan_type"] == "H"].copy()
    historical_df, application_df = partition_data(hist_only_df, valid_mapping)

    assert len(historical_df) == 3
    assert len(application_df) == 0

def test_partition_missing_loan_type_column(sample_df, valid_mapping):
    """Test error when the column specified by 'loan_type' is missing from DataFrame."""
    df_no_type = sample_df.drop(columns=["loan_type"])
    with pytest.raises(ValueError, match=r"The column 'loan_type' \(mapped to 'loan_type'\) was not found in the DataFrame."):
        partition_data(df_no_type, valid_mapping)

def test_partition_missing_loan_type_in_mapping(sample_df):
    """Test error when 'loan_type' is not specified in the mapping config."""
    mapping_no_loan_type_field = ColumnMapping(
        loan_id="loan_id",
        # loan_type="loan_type", # This is missing
        default_flag="default_flag",
        exposure="exposure",
        loan_type_mapping={"historical": "H", "application": "A"},
    )
    with pytest.raises(ValueError, match="The 'loan_type' field in 'column_mapping' must be specified"):
        partition_data(sample_df, mapping_no_loan_type_field)


def test_partition_incomplete_mapping_in_config(sample_df):
    """Test error when 'loan_type_mapping' is incomplete."""
    # Test with a valid mapping but trigger the error in the function
    mapping_missing_key = ColumnMapping(
        loan_id="loan_id",
        loan_type="loan_type",
        default_flag="default_flag",
        exposure="exposure",
        loan_type_mapping={"history": "H", "app": "A"}, # 'historical' key is misspelled
    )
    with pytest.raises(ValueError, match="must contain keys for both 'historical' and 'application'"):
        partition_data(sample_df, mapping_missing_key)

def test_partition_data_with_different_identifiers(valid_mapping):
    """Test that custom identifiers from the config are respected."""
    data = {
        "loan_id": [1, 2, 3, 4],
        "loan_type": ["Old", "New", "Old", "New"],
        "default_flag": [1, 0, 0, 0],
    }
    df = pd.DataFrame(data)
    
    mapping = ColumnMapping(
        loan_id="loan_id",
        loan_type="loan_type",
        default_flag="default_flag",
        exposure="exposure",
        loan_type_mapping={"historical": "Old", "application": "New"},
    )
    
    historical_df, application_df = partition_data(df, mapping)
    assert len(historical_df) == 2
    assert len(application_df) == 2
    assert list(historical_df['loan_id']) == [1, 3]
    assert list(application_df['loan_id']) == [2, 4]
