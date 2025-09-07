"""
Example: Loading and validating a YAML config file (Feature 1.3.1)
"""
from irbstudio.data.loader import load_config

config = load_config("examples/sample_config.yaml")

print("Loaded config from YAML:")
print(config)
