import pandas as pd
import pytest
from irbstudio.data.loader import load_portfolio
from irbstudio.config.schema import ColumnMapping

def test_load_portfolio_missing_column_raises_error(tmp_path):
    """
    Tests that a clear error is raised when a required column is missing
    after mapping.
    """
    # Create a sample portfolio with a missing required column ('balance' -> 'exposure')
    data = {'loan_identifier': ['A1', 'B2'], 'ltv': [0.8, 0.9]}
    df = pd.DataFrame(data)
    p = tmp_path / "missing_col_portfolio.csv"
    df.to_csv(p, index=False)

    # Mapping expects 'balance' but it's not in the file
    mapping = ColumnMapping(loan_id="loan_identifier", exposure="balance")

    with pytest.raises(ValueError) as excinfo:
        load_portfolio(str(p), mapping)

    # Check for the enhanced error message
    assert "Portfolio is missing required columns after mapping: ['exposure']" in str(excinfo.value)
    assert "Original columns found in the file: ['loan_identifier', 'ltv']" in str(excinfo.value)
