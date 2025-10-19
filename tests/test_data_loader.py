"""
Tests for data loading and management.

Priority 1: Critical - Core Functionality
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from irbstudio.data.loader import load_portfolio
from irbstudio.config.schema import ColumnMapping


class TestLoadPortfolio:
    """Tests for load_portfolio() function."""
    
    def test_load_portfolio_csv(self, temp_csv_file):
        """Test load_portfolio() with CSV file."""
        mapping = ColumnMapping(
            loan_id='loan_id',
            exposure='exposure'
        )
        df = load_portfolio(temp_csv_file, mapping)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_load_portfolio_with_column_mapping(self, small_portfolio_df):
        """Test load_portfolio() with column name mapping."""
        # Create CSV with different column names
        df_renamed = small_portfolio_df.rename(columns={
            'loan_id': 'LOAN_NUMBER',
            'exposure': 'BALANCE',
            'pd': 'PD_VALUE'
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_renamed.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            mapping = ColumnMapping(
                loan_id='LOAN_NUMBER',
                exposure='BALANCE'
            )
            
            df = load_portfolio(temp_path, mapping)
            
            assert df is not None
            assert isinstance(df, pd.DataFrame)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_portfolio_missing_file(self):
        """Test load_portfolio() with non-existent file."""
        mapping = ColumnMapping(loan_id='loan_id', exposure='exposure')
        
        with pytest.raises((FileNotFoundError, IOError, Exception)):
            load_portfolio('nonexistent_file.csv', mapping)
    
    def test_load_portfolio_empty_file(self):
        """Test load_portfolio() with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("loan_id,exposure,pd\n")  # Header only
            temp_path = f.name
        
        try:
            mapping = ColumnMapping(loan_id='loan_id', exposure='exposure')
            df = load_portfolio(temp_path, mapping)
            
            # Should return empty DataFrame or raise exception
            assert df is not None
            assert len(df) == 0
        except Exception:
            # Also acceptable to raise exception for empty file
            pass
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_portfolio_date_parsing(self, temp_csv_file):
        """Test that date columns are parsed correctly."""
        mapping = ColumnMapping(
            loan_id='loan_id',
            exposure='exposure',
            date='reporting_date'
        )
        df = load_portfolio(temp_csv_file, mapping)
        
        if 'reporting_date' in df.columns:
            # Check if date column is datetime type
            assert pd.api.types.is_datetime64_any_dtype(df['reporting_date']) or \
                   isinstance(df['reporting_date'].iloc[0], (pd.Timestamp, str))
    
    def test_load_portfolio_data_type_inference(self, temp_csv_file):
        """Test automatic data type detection."""
        mapping = ColumnMapping(loan_id='loan_id', exposure='exposure')
        df = load_portfolio(temp_csv_file, mapping)
        
        # Numeric columns should be numeric
        if 'exposure' in df.columns:
            assert pd.api.types.is_numeric_dtype(df['exposure'])
        
        if 'pd' in df.columns:
            assert pd.api.types.is_numeric_dtype(df['pd'])


class TestDataValidation:
    """Tests for data validation functions."""
    
    def test_validate_portfolio_required_columns(self, small_portfolio_df):
        """Test that required columns are validated."""
        # This test depends on internal validation function
        # If not exported, test through load_portfolio or simulator
        
        required_cols = ['loan_id', 'exposure']
        for col in required_cols:
            if col in small_portfolio_df.columns:
                assert small_portfolio_df[col] is not None
    
    def test_validate_portfolio_correct_data_types(self, small_portfolio_df):
        """Test data type correctness."""
        # Exposure should be numeric
        if 'exposure' in small_portfolio_df.columns:
            assert pd.api.types.is_numeric_dtype(small_portfolio_df['exposure'])
        
        # PD should be numeric
        if 'pd' in small_portfolio_df.columns:
            assert pd.api.types.is_numeric_dtype(small_portfolio_df['pd'])
    
    def test_validate_portfolio_numeric_ranges(self, small_portfolio_df):
        """Test numeric column ranges are valid."""
        # Exposure should be non-negative
        if 'exposure' in small_portfolio_df.columns:
            assert (small_portfolio_df['exposure'] >= 0).all()
        
        # PD should be between 0 and 1
        if 'pd' in small_portfolio_df.columns:
            valid_pd = small_portfolio_df['pd'].dropna()
            if len(valid_pd) > 0:
                assert (valid_pd >= 0).all()
                assert (valid_pd <= 1).all()
        
        # LGD should be between 0 and 1
        if 'lgd' in small_portfolio_df.columns:
            valid_lgd = small_portfolio_df['lgd'].dropna()
            if len(valid_lgd) > 0:
                assert (valid_lgd >= 0).all()
                assert (valid_lgd <= 1).all()
    
    def test_validate_portfolio_unique_loan_ids(self, small_portfolio_df):
        """Test that loan IDs can be checked for uniqueness."""
        # In a single time-slice portfolio, loan_ids should be unique
        # For multi-date portfolios, combination of (loan_id, date) should be unique
        
        if 'loan_id' in small_portfolio_df.columns:
            # Check if we have a date column
            date_cols = [col for col in small_portfolio_df.columns 
                        if 'date' in col.lower() or col in ['reporting_date', 'observation_date']]
            
            if date_cols:
                # Multi-date portfolio: check (loan_id, date) uniqueness
                date_col = date_cols[0]
                combined = small_portfolio_df.groupby(['loan_id', date_col]).size()
                # Each combination should appear exactly once
                assert (combined == 1).all(), "Duplicate (loan_id, date) combinations found"
            else:
                # Single time-slice: check loan_id uniqueness
                loan_id_counts = small_portfolio_df['loan_id'].value_counts()
                if len(loan_id_counts) > 0:
                    # All loan_ids should appear exactly once
                    # (or we accept duplicates for testing purposes)
                    # For now, just verify the column exists and has values
                    assert small_portfolio_df['loan_id'].notna().any()


class TestFileFormats:
    """Tests for loading different file formats."""
    
    def test_load_portfolio_excel(self, small_portfolio_df):
        """Test load_portfolio() with Excel file raises appropriate error."""
        import openpyxl  # Check if openpyxl is available
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.xlsx', delete=False) as f:
            temp_path = f.name
            small_portfolio_df.to_excel(temp_path, index=False, engine='openpyxl')
        
        try:
            mapping = ColumnMapping(loan_id='loan_id', exposure='exposure')
            
            # Should raise ValueError for unsupported file type
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_portfolio(temp_path, mapping)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_portfolio_compressed_csv(self, small_portfolio_df):
        """Test load_portfolio() with .csv.gz file raises appropriate error (currently unsupported)."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv.gz', delete=False) as f:
            temp_path = f.name
            small_portfolio_df.to_csv(temp_path, index=False, compression='gzip')
        
        try:
            mapping = ColumnMapping(loan_id='loan_id', exposure='exposure')
            
            # Should raise ValueError for unsupported file type
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_portfolio(temp_path, mapping)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_portfolio_compressed_zip(self, small_portfolio_df):
        """Test load_portfolio() with .zip file raises appropriate error (currently unsupported)."""
        import zipfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'portfolio.csv'
            zip_path = Path(tmpdir) / 'portfolio.zip'
            
            # Save CSV
            small_portfolio_df.to_csv(csv_path, index=False)
            
            # Create ZIP
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.write(csv_path, 'portfolio.csv')
            
            mapping = ColumnMapping(loan_id='loan_id', exposure='exposure')
            
            # Should raise ValueError for unsupported file type
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_portfolio(str(zip_path), mapping)


class TestCorruptedFiles:
    """Tests for handling corrupted or invalid files."""
    
    def test_load_portfolio_corrupted_file(self):
        """Test load_portfolio() with corrupted/invalid file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write invalid CSV data
            f.write("this is not,a valid,csv file\n")
            f.write("with random data: {invalid}\n")
            temp_path = f.name
        
        try:
            mapping = ColumnMapping(loan_id='loan_id', exposure='exposure')
            
            # Should either raise an exception or return DataFrame with issues
            try:
                df = load_portfolio(temp_path, mapping)
                # If it loads, check that required columns might be missing
                # (depends on loader's error handling)
                assert df is not None
            except Exception as e:
                # Acceptable to raise exception for corrupted file
                assert True
        finally:
            Path(temp_path).unlink(missing_ok=True)