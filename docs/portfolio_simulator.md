# PortfolioSimulator Class Implementation

## Overview

The `PortfolioSimulator` class provides an object-oriented implementation of the portfolio simulation engine, replacing the procedural implementation in `pd_simulator.py`. This implementation follows the refactoring outlined in Task 2.3.1 of the project plan.

## Key Features

1. **Separation of Preparation and Simulation**:
   - Preparation is deterministic and done once: `prepare_simulation()`
   - Simulation is stochastic and can be repeated: `simulate_once()`

2. **Monte Carlo Simulation**:
   - Added built-in support for Monte Carlo simulations: `run_monte_carlo()`
   - Efficient reuse of fitted models across simulations

3. **Improved Random Seed Handling**:
   - Random seed can be set at the class level or for individual simulations
   - Consistent reproducible results with the same seed

4. **Backward Compatibility**:
   - Added `simulate_portfolio()` function that behaves identically to the original

## Implementation Details

The implementation consists of:

1. **Class Initialization**:
   - Stores all configuration parameters
   - Sets random seed if provided
   - Initializes placeholders for simulation components

2. **Preparation Stage**:
   - Segments the portfolio into historical and application samples
   - Fits Beta Mixture Model to historical scores
   - Infers systemic factors
   - Calculates migration matrices

3. **Simulation Stage**:
   - Simulates ratings for existing clients using migration matrices
   - Generates scores for new clients using fitted distributions
   - Applies appropriate PD mappings to simulated ratings

4. **Monte Carlo Support**:
   - Runs multiple simulations with unique seeds derived from a base seed
   - Returns a list of simulated portfolios

## Testing

The implementation has been tested with a comprehensive suite of unit tests:

1. **Initialization Test**:
   - Verifies that all parameters are correctly stored

2. **Preparation Test**:
   - Verifies that all required components are generated

3. **Compatibility Test**:
   - Verifies that the new implementation produces equivalent results to the original

4. **Monte Carlo Test**:
   - Verifies that multiple simulations produce different results

5. **Reproducibility Test**:
   - Verifies that the same random seed produces identical results

## Usage Example

```python
from irbstudio.simulation.portfolio_simulator import PortfolioSimulator

# Create simulator instance
simulator = PortfolioSimulator(
    portfolio_df=portfolio_df,
    score_to_rating_bounds=score_to_rating_bounds,
    rating_col='rating',
    loan_id_col='loan_id',
    date_col='date',
    default_col='is_default',
    into_default_flag_col='into_default',
    score_col='score',
    application_start_date=application_start_date,
    target_auc=0.75,
    random_seed=42
)

# Basic simulation (prepare and run once)
simulated_portfolio = simulator.prepare_simulation().simulate_once()

# Monte Carlo simulation (run multiple times)
monte_carlo_results = simulator.run_monte_carlo(num_iterations=100)
```

## Backward Compatibility

The original `simulate_portfolio` function is still available and works identically to the previous implementation:

```python
from irbstudio.simulation.portfolio_simulator import simulate_portfolio

# Use exactly the same as before
simulated_portfolio = simulate_portfolio(
    portfolio_df=portfolio_df,
    score_to_rating_bounds=score_to_rating_bounds,
    rating_col='rating',
    loan_id_col='loan_id',
    date_col='date',
    default_col='is_default',
    into_default_flag_col='into_default',
    score_col='score',
    application_start_date=application_start_date,
    target_auc=0.75
)
```

## Future Improvements

1. **Parallel Processing**: Add support for parallel execution of Monte Carlo simulations
2. **Progress Tracking**: Add a progress callback for long-running simulations
3. **Visualization**: Add built-in visualization methods for simulation results