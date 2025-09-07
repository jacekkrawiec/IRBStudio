"""
Example: Loading and mapping a portfolio file using IRBStudio (Feature 1.3.2)
"""
from irbstudio.data import load_config, load_portfolio

# Load config from YAML
config = load_config("examples/sample_config.yaml")

# Load and map the portfolio CSV
portfolio = load_portfolio("examples/sample_portfolio.csv", config.column_mapping)

print("Standardized portfolio DataFrame:")
print(portfolio)
