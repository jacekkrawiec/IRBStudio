"""
Advanced Data Management tests (Priority 2).

This module contains advanced tests for data loading, validation, and management:
- Large file handling
- Multiple file formats
- Column mapping edge cases
- Data validation
- Missing data handling
- Date parsing and formatting

Tests are organized by functional area.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
from irbstudio.data.loader import load_portfolio, validate_portfolio, load_config
from irbstudio.config.schema import ColumnMapping, Config


class TestLargeFileHandling:
    """Tests for handling large portfolio files."""
    
    def test_load_large_csv_file(self):
        """Test loading a large CSV file (simulated with chunking)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            # Write header
            f.write('loan_id,balance,pd,rating,date,default\n')
            
            # Write many rows (simulate large file)
            for i in range(10000):
                f.write(f'L{i},{100000 + i},{0.01 + i*0.0001},A,2024-01-01,0\n')
            
            temp_path = f.name
        
        try:
            # Load with minimal mapping
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='balance',  # balance maps to exposure (canonical name)
                pd='pd',
                rating='rating',
                date='date',
                default='default'
            )
            
            df = load_portfolio(temp_path, mapping)
            
            assert len(df) == 10000
            assert 'loan_id' in df.columns
            assert 'exposure' in df.columns  # Canonical name
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.skip(reason="Parquet support requires pyarrow or fastparquet")
    def test_load_parquet_file(self):
        """Test loading Parquet format files."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'loan_id': [f'L{i}' for i in range(1000)],
            'balance': np.random.uniform(50000, 500000, 1000),
            'pd': np.random.uniform(0.01, 0.10, 1000),
            'rating': np.random.choice(['A', 'B', 'C'], 1000),
            'date': pd.to_datetime('2024-01-01'),
            'default': np.random.choice([0, 1], 1000, p=[0.98, 0.02])
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save as parquet
            test_df.to_parquet(temp_path, index=False)
            
            # Load it back
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='balance',
                pd='pd',
                rating='rating',
                date='date',
                default='default'
            )
            
            df = load_portfolio(temp_path, mapping)
            
            assert len(df) == 1000
            assert list(df.columns) == list(test_df.columns)
        finally:
            os.unlink(temp_path)
    
    def test_load_file_with_pathlib(self):
        """Test loading files using pathlib.Path objects."""
        test_df = pd.DataFrame({
            'loan_id': ['L1', 'L2', 'L3'],
            'balance': [100000, 200000, 150000],
            'pd': [0.01, 0.02, 0.015],
            'rating': ['A', 'B', 'A'],
            'date': pd.to_datetime('2024-01-01'),
            'default': [0, 0, 0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            test_df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            # Load using Path object
            path_obj = Path(temp_path)
            
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='balance',
                pd='pd',
                rating='rating',
                date='date',
                default='default'
            )
            
            df = load_portfolio(path_obj, mapping)
            
            assert len(df) == 3
            assert 'loan_id' in df.columns
        finally:
            os.unlink(temp_path)


class TestColumnMappingAdvanced:
    """Advanced tests for column mapping."""
    
    def test_load_with_complex_mapping(self):
        """Test loading with complex column name mapping."""
        # Create file with non-standard column names
        test_df = pd.DataFrame({
            'LOAN_IDENTIFIER': ['L1', 'L2', 'L3'],
            'OUTSTANDING_BALANCE': [100000, 200000, 150000],
            'PROBABILITY_DEFAULT': [0.01, 0.02, 0.015],
            'CREDIT_RATING': ['A', 'B', 'A'],
            'REPORTING_DATE': pd.to_datetime('2024-01-01'),
            'DEFAULT_FLAG': [0, 0, 0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            test_df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            # Map non-standard names to canonical names
            mapping = ColumnMapping(
                loan_id='LOAN_IDENTIFIER',
                exposure='OUTSTANDING_BALANCE',  # exposure is canonical name
                pd='PROBABILITY_DEFAULT',
                rating='CREDIT_RATING',
                date='REPORTING_DATE',
                default='DEFAULT_FLAG'
            )
            
            df = load_portfolio(temp_path, mapping)
            
            # Verify canonical names are used for mapped columns
            assert 'loan_id' in df.columns
            assert 'exposure' in df.columns  # Canonical name
            assert 'rating' in df.columns
            assert 'date' in df.columns
            
            # Original names should be gone for mapped columns
            assert 'LOAN_IDENTIFIER' not in df.columns
            assert 'OUTSTANDING_BALANCE' not in df.columns
            assert 'CREDIT_RATING' not in df.columns
            assert 'REPORTING_DATE' not in df.columns
        finally:
            os.unlink(temp_path)
    
    def test_load_with_optional_columns(self):
        """Test loading with optional columns present."""
        test_df = pd.DataFrame({
            'loan_id': ['L1', 'L2', 'L3'],
            'balance': [100000, 200000, 150000],
            'pd': [0.01, 0.02, 0.015],
            'rating': ['A', 'B', 'A'],
            'date': pd.to_datetime('2024-01-01'),
            'default': [0, 0, 0],
            'ltv': [0.75, 0.80, 0.70],  # Optional
            'property_value': [133333, 250000, 214286]  # Optional
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            test_df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='balance',
                pd='pd',
                rating='rating',
                date='date',
                default='default',
                ltv='ltv',
                property_value='property_value'
            )
            
            df = load_portfolio(temp_path, mapping)
            
            # Both required and optional columns should be present
            assert 'ltv' in df.columns
            assert 'property_value' in df.columns
            assert len(df) == 3
        finally:
            os.unlink(temp_path)
    
    def test_load_with_missing_optional_columns(self):
        """Test that missing optional columns don't cause errors."""
        test_df = pd.DataFrame({
            'loan_id': ['L1', 'L2', 'L3'],
            'balance': [100000, 200000, 150000],
            'pd': [0.01, 0.02, 0.015],
            'rating': ['A', 'B', 'A'],
            'date': pd.to_datetime('2024-01-01'),
            'default': [0, 0, 0]
            # No optional columns
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            test_df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='balance',
                pd='pd',
                rating='rating',
                date='date',
                default='default'
                # No mapping for optional columns
            )
            
            df = load_portfolio(temp_path, mapping)
            
            # Should load successfully without optional columns
            assert len(df) == 3
            assert 'loan_id' in df.columns
        finally:
            os.unlink(temp_path)


class TestDataValidation:
    """Tests for data validation."""
    
    def test_validate_portfolio_missing_required_columns(self):
        """Test that validation fails when required columns are missing."""
        # DataFrame missing 'exposure' column (required)
        df = pd.DataFrame({
            'loan_id': ['L1', 'L2'],
            'pd': [0.01, 0.02],
            'date': pd.to_datetime('2024-01-01'),
            'default': [0, 0]
            # Missing 'exposure' (required)
        })
        
        original_columns = df.columns.tolist()
        
        with pytest.raises(ValueError, match="missing required columns"):
            validate_portfolio(df, original_columns)
    
    def test_validate_portfolio_with_all_required_columns(self):
        """Test that validation passes with all required columns."""
        df = pd.DataFrame({
            'loan_id': ['L1', 'L2'],
            'exposure': [100000, 200000],
            'pd': [0.01, 0.02],
            'rating': ['A', 'B'],
            'date': pd.to_datetime('2024-01-01'),
            'default': [0, 0]
        })
        
        original_columns = df.columns.tolist()
        
        # Should not raise
        validate_portfolio(df, original_columns)
    
    def test_load_unsupported_file_format(self):
        """Test that unsupported file formats raise appropriate error."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        
        try:
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='balance',
                pd='pd',
                rating='rating',
                date='date',
                default='default'
            )
            
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_portfolio(temp_path, mapping)
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        mapping = ColumnMapping(
            loan_id='loan_id',
            exposure='balance',
            pd='pd',
            rating='rating',
            date='date',
            default='default'
        )
        
        with pytest.raises(FileNotFoundError):
            load_portfolio('/nonexistent/path/file.csv', mapping)


class TestDateHandling:
    """Tests for date parsing and handling."""
    
    def test_load_with_various_date_formats(self):
        """Test loading files with different date formats."""
        test_df = pd.DataFrame({
            'loan_id': ['L1', 'L2', 'L3'],
            'balance': [100000, 200000, 150000],
            'pd': [0.01, 0.02, 0.015],
            'rating': ['A', 'B', 'A'],
            'date': ['2024-01-01', '2024-02-01', '2024-03-01'],  # String dates
            'default': [0, 0, 0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            test_df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='balance',
                pd='pd',
                rating='rating',
                date='date',
                default='default'
            )
            
            df = load_portfolio(temp_path, mapping)
            
            # Dates should be loaded (as strings or datetime)
            assert 'date' in df.columns
            assert len(df) == 3
        finally:
            os.unlink(temp_path)
    
    def test_load_with_timestamp_dates(self):
        """Test loading files with timestamp dates."""
        test_df = pd.DataFrame({
            'loan_id': ['L1', 'L2', 'L3'],
            'balance': [100000, 200000, 150000],
            'pd': [0.01, 0.02, 0.015],
            'rating': ['A', 'B', 'A'],
            'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'default': [0, 0, 0]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            test_df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            mapping = ColumnMapping(
                loan_id='loan_id',
                exposure='balance',
                pd='pd',
                rating='rating',
                date='date',
                default='default'
            )
            
            df = load_portfolio(temp_path, mapping)
            
            assert 'date' in df.columns
            assert len(df) == 3
        finally:
            os.unlink(temp_path)


class TestConfigLoading:
    """Tests for configuration loading."""
    
    def test_load_config_with_pathlib(self):
        """Test loading config using pathlib.Path object."""
        config_dict = {
            'portfolio_path': 'data.csv',
            'column_mapping': {
                'loan_id': 'loan_id',
                'exposure': 'balance',
                'pd': 'pd',
                'rating': 'rating',
                'date': 'date',
                'default': 'default'
            },
            'score_to_rating_bounds': {
                'A': [0.0, 0.3],
                'B': [0.3, 0.7],
                'C': [0.7, 1.0]
            },
            'scenarios': [{
                'name': 'baseline',
                'pd_auc': 0.75,
                'portfolio_default_rate': 0.02,
                'lgd': 0.25
            }],
            'simulators': [{
                'name': 'baseline',
                'n_iterations': 10
            }],
            'calculators': [{
                'type': 'AIRB',
                'name': 'AIRB_calc',
                'params': {
                    'asset_correlation': 0.15,
                    'lgd': 0.25
                }
            }]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            # Load using Path object
            path_obj = Path(temp_path)
            config = load_config(path_obj)
            
            assert config is not None
            assert isinstance(config, Config)
            # Config doesn't have portfolio_path, just scenarios
            assert len(config.scenarios) > 0
            assert config.scenarios[0].name == 'baseline'
        finally:
            os.unlink(temp_path)


# Summary of test coverage:
# - Large file handling: 3 tests
# - Column mapping advanced: 4 tests
# - Data validation: 4 tests
# - Date handling: 2 tests
# - Config loading: 1 test
# Total: 14 tests for advanced data management
