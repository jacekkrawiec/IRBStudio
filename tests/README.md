# IRBStudio Test Suite

This directory contains the test suite for IRBStudio, organized by priority and module.

## 📁 Test Structure

```
tests/
├── conftest.py                      # Pytest fixtures and test utilities
├── test_main_api.py                 # High-level API tests (Priority 1)
├── test_rwa_calculators.py          # AIRB and SA calculator tests (Priority 1)
├── test_portfolio_simulator.py      # Monte Carlo simulation tests (Priority 1)
├── test_integrated_analysis.py      # Scenario analysis tests (Priority 1)
├── test_data_loader.py              # Data loading tests (Priority 1)
├── test_config_schema.py            # Configuration schema tests (Priority 1)
└── README.md                        # This file
```

## 🚀 Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_rwa_calculators.py
```

### Run Specific Test Class
```bash
pytest tests/test_rwa_calculators.py::TestAIRBMortgageCalculator
```

### Run Specific Test
```bash
pytest tests/test_rwa_calculators.py::TestAIRBMortgageCalculator::test_airb_calculator_init
```

### Run with Verbose Output
```bash
pytest -v
```

### Run with Coverage Report
```bash
pytest --cov=irbstudio --cov-report=html
```

### Run Only Priority 1 Tests
```bash
pytest tests/test_main_api.py tests/test_rwa_calculators.py tests/test_portfolio_simulator.py tests/test_integrated_analysis.py tests/test_data_loader.py tests/test_config_schema.py
```

## 📊 Test Categories

### Priority 1: Critical - Core Functionality ✅
These tests cover the essential functionality required for IRBStudio to work:

- **test_main_api.py** (18 tests)
  - `run_analysis()` function
  - `run_scenario_comparison()` function
  - `load_config()` function

- **test_rwa_calculators.py** (30+ tests)
  - AIRBMortgageCalculator
  - SAMortgageCalculator
  - RWAResult class

- **test_portfolio_simulator.py** (20+ tests)
  - PortfolioSimulator initialization
  - prepare_simulation() method
  - simulate_once() method
  - run_monte_carlo() method
  - Beta Mixture Model

- **test_integrated_analysis.py** (15+ tests)
  - IntegratedAnalysis orchestration
  - Calculator management
  - Scenario management
  - run_scenario() method
  - Statistical summaries

- **test_data_loader.py** (10+ tests)
  - load_portfolio() function
  - Data validation
  - Column mapping
  - Data type inference

- **test_config_schema.py** (15+ tests)
  - Config schema validation
  - Scenario schema
  - ColumnMapping schema
  - RegulatoryParams schema

**Total Priority 1 Tests: ~108**

### Priority 2: Important - Key Features (To Be Implemented)
- Scenario comparison tests
- Date breakdown tests
- Memory-efficient processing tests
- Visualization tests

### Priority 3: Nice to Have - Advanced/Edge Cases (To Be Implemented)
- Performance tests
- Edge case tests
- Progress tracking tests
- Logging tests

## 🧪 Test Fixtures

### Portfolio Data Fixtures
- `sample_portfolio_df` - 1,000 loan portfolio with multiple dates
- `small_portfolio_df` - 100 loan portfolio for quick tests
- `multi_date_portfolio_df` - Portfolio with 12 monthly reporting dates

### Configuration Fixtures
- `sample_config_dict` - Standard configuration dictionary
- `temp_config_file` - Temporary YAML config file
- `temp_csv_file` - Temporary CSV portfolio file
- `temp_output_dir` - Temporary output directory

### Parameter Fixtures
- `airb_params` - Standard AIRB calculator parameters
- `sa_params` - Standard SA calculator parameters
- `column_mapping` - Standard column mapping

### Helper Functions
- `assert_dataframe_structure()` - Validate DataFrame structure
- `assert_numeric_column()` - Validate numeric columns
- `assert_positive_values()` - Validate non-negative values
- `assert_in_range()` - Validate value ranges

## 📋 Test Checklist

### Currently Implemented
- ✅ High-Level API tests (18 tests)
- ✅ RWA Calculator tests (30+ tests)
- ✅ Portfolio Simulator tests (20+ tests)
- ✅ Integrated Analysis tests (15+ tests)
- ✅ Data Loader tests (10+ tests)
- ✅ Config Schema tests (15+ tests)

### To Be Implemented
- ⏳ Reporting & Visualization tests (34 tests)
- ⏳ Advanced Features tests (32 tests)
- ⏳ Utility Functions tests (16 tests)
- ⏳ Integration tests (13 tests)
- ⏳ Performance tests (14 tests)
- ⏳ Edge Case tests (34 tests)

## 🎯 Test Coverage Goals

- **Line Coverage:** ≥ 80%
- **Branch Coverage:** ≥ 70%
- **Critical Paths:** 100%

### Current Coverage by Module
```
Module                              Coverage
─────────────────────────────────────────────
irbstudio.main                      ≥ 80%
irbstudio.engine.mortgage           ≥ 80%
irbstudio.simulation               ≥ 75%
irbstudio.engine.integrated_analysis ≥ 75%
irbstudio.data.loader              ≥ 80%
irbstudio.config.schema            ≥ 80%
```

## 🐛 Debugging Failed Tests

### View Test Output
```bash
pytest -v -s
```

### Run Last Failed Tests Only
```bash
pytest --lf
```

### Drop into Debugger on Failure
```bash
pytest --pdb
```

### Generate HTML Report
```bash
pytest --html=report.html --self-contained-html
```

## 📝 Writing New Tests

### Test Naming Convention
```python
def test_<module>_<function>_<scenario>():
    """Test description."""
    # Arrange
    # ... setup test data
    
    # Act
    # ... execute function
    
    # Assert
    # ... verify results
```

### Example Test
```python
def test_airb_calculate_rw_basic(self, airb_params, small_portfolio_df):
    """Test calculate_rw() basic risk weight calculation."""
    # Arrange
    calculator = AIRBMortgageCalculator(airb_params)
    
    # Act
    result_df = calculator.calculate_rw(small_portfolio_df)
    
    # Assert
    assert 'risk_weight' in result_df.columns
    assert (result_df['risk_weight'] >= 0).all()
```

## 🔧 Continuous Integration

Tests are automatically run on:
- Every commit to `main` branch
- Every pull request
- Nightly build

### CI Requirements
- All Priority 1 tests must pass
- Code coverage must not decrease
- No new linting errors

## 📚 References

- [pytest Documentation](https://docs.pytest.org/)
- [IRBStudio Test List](../docs/tests_list.md)
- [IRBStudio Features](../docs/available_features.md)

---

**Last Updated:** October 19, 2025  
**Maintained By:** IRBStudio Team
