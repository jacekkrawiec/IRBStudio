import pandas as pd
from irbstudio.data.loader import load_portfolio
from irbstudio.config.schema import ColumnMapping


def test_load_portfolio_csv():
    mapping = ColumnMapping(
        loan_id="loan_identifier",
        exposure="balance",
        ltv="ltv"
    )
    df = load_portfolio("tests/data/sample_portfolio.csv", mapping)
    # Check columns are mapped to canonical names
    assert set(["loan_id", "exposure", "ltv"]).issubset(df.columns)
    # Check data is loaded correctly
    assert df.loc[0, "loan_id"] == "A1"
    assert df.loc[1, "exposure"] == 200000
    assert df.loc[0, "ltv"] == 0.8
