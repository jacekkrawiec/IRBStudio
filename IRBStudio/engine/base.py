# Abstract base classes for calculators

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional, List


class BaseRWACalculator(ABC):
    """
    Abstract base class for all RWA calculators.
    
    This class defines the interface that all RWA calculator implementations must follow.
    Different approaches (AIRB, SA) and asset classes (mortgage, corporate, etc.) will
    have their own implementations of this interface.
    """
    
    def __init__(self, regulatory_params: Dict[str, Any]):
        """
        Initialize the calculator with regulatory parameters.
        
        Args:
            regulatory_params: A dictionary of regulatory parameters that influence
                              the RWA calculation, such as asset correlation, 
                              confidence level, etc.
        """
        self.regulatory_params = regulatory_params
    
    @abstractmethod
    def calculate_rw(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk weights for each exposure in the portfolio.
        
        Args:
            portfolio_df: DataFrame containing the portfolio data.
                         Must include required fields specific to the calculator.
                         
        Returns:
            DataFrame with original data plus a 'risk_weight' column.
        """
        pass
    
    @abstractmethod
    def calculate_rwa(self, 
                     portfolio_df: pd.DataFrame, 
                     date_column: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate RWA for each exposure in the portfolio.
        
        This typically calls calculate_rw() first and then multiplies by exposure.
        
        Args:
            portfolio_df: DataFrame containing the portfolio data.
                         Must include required fields specific to the calculator.
            date_column: Optional name of the date column to handle calculations
                         by date if multiple dates are present.
                         
        Returns:
            DataFrame with original data plus 'risk_weight' and 'rwa' columns.
        """
        pass
    
    def validate_inputs(self, portfolio_df: pd.DataFrame, required_columns: list) -> None:
        """
        Validate that the portfolio DataFrame contains all required columns.
        
        Args:
            portfolio_df: DataFrame to validate.
            required_columns: List of column names that must be present.
            
        Raises:
            ValueError: If any required column is missing.
        """
        missing_columns = [col for col in required_columns if col not in portfolio_df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Portfolio DataFrame must contain: {required_columns}"
            )
    
    def summarize_rwa(self, 
                    portfolio_df: pd.DataFrame, 
                    breakdown_fields: Optional[List[str]] = None,
                    date_field: Optional[str] = None) -> Dict[str, Any]:
        """
        Provide a summary of RWA calculation results.
        
        Args:
            portfolio_df: DataFrame with calculated risk weights and RWA.
            breakdown_fields: Optional list of fields to include in breakdown summaries
                             (e.g., 'rating', 'segment', 'product_type').
            date_field: Optional name of date field for time-based breakdowns.
                        If specified and multiple dates exist, results will be
                        summarized by date.
            
        Returns:
            Dictionary with summary statistics such as total RWA, average risk weight, etc.
            If breakdown_fields are provided, includes nested dictionaries with breakdowns.
            If date_field is provided and multiple dates exist, includes time-based summaries.
        """
        if 'rwa' not in portfolio_df.columns:
            raise ValueError("RWA must be calculated before summarizing.")
        
        # Default result with overall metrics
        result = {
            'total_rwa': portfolio_df['rwa'].sum(),
            'average_risk_weight': portfolio_df['risk_weight'].mean(),
            'total_exposure': portfolio_df['exposure'].sum(),
            'weighted_average_rw': (portfolio_df['rwa'].sum() / portfolio_df['exposure'].sum())
        }
        
        # Process date-based breakdown if requested and multiple dates exist
        if date_field and date_field in portfolio_df.columns:
            dates = portfolio_df[date_field].unique()
            if len(dates) > 1:
                result['by_date'] = {}
                for date in dates:
                    date_df = portfolio_df[portfolio_df[date_field] == date]
                    date_key = str(date)  # Convert to string for JSON compatibility
                    result['by_date'][date_key] = {
                        'total_rwa': date_df['rwa'].sum(),
                        'average_risk_weight': date_df['risk_weight'].mean(),
                        'total_exposure': date_df['exposure'].sum(),
                        'weighted_average_rw': (date_df['rwa'].sum() / date_df['exposure'].sum())
                    }
        
        # Add user-specified breakdowns
        if breakdown_fields:
            for field in breakdown_fields:
                if field in portfolio_df.columns:
                    field_key = f'by_{field}'
                    result[field_key] = {}
                    
                    # Group by the breakdown field and calculate summary metrics
                    breakdown_df = portfolio_df.groupby(field).agg({
                        'exposure': 'sum',
                        'rwa': 'sum',
                        'risk_weight': 'mean'
                    })
                    
                    # Add weighted average calculation
                    breakdown_df['weighted_average_rw'] = breakdown_df['rwa'] / breakdown_df['exposure']
                    
                    # Convert to dictionary for JSON serialization
                    result[field_key] = breakdown_df.to_dict()
        
        # For backwards compatibility, always include rating breakdown if available
        if 'rating' in portfolio_df.columns and not breakdown_fields:
            result['rwa_by_rating'] = portfolio_df.groupby('rating').agg({
                'exposure': 'sum',
                'rwa': 'sum'
            }).to_dict()
            
        return result


class RWAResult:
    """
    Class to hold the results of an RWA calculation.
    
    This provides a standard structure for returning results from any calculator,
    making it easier to process and compare results from different approaches.
    Supports breakdowns by rating, segment, date, or any other fields specified
    in the summary dictionary.
    """
    
    def __init__(self, 
                 portfolio_with_rwa: pd.DataFrame,
                 summary: Dict[str, Any],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize with calculation results.
        
        Args:
            portfolio_with_rwa: DataFrame with risk weights and RWA.
            summary: Dictionary with summary statistics. Can include nested
                    dictionaries with breakdowns by rating, segment, date, etc.
            metadata: Optional metadata about the calculation.
        """
        self.portfolio = portfolio_with_rwa
        self.summary = summary
        self.metadata = metadata or {}
    
    @property
    def total_rwa(self) -> float:
        """Get the total RWA."""
        return self.summary['total_rwa']
    
    @property
    def total_exposure(self) -> float:
        """Get the total exposure."""
        return self.summary['total_exposure']
    
    @property
    def capital_requirement(self) -> float:
        """Calculate the capital requirement (8% of RWA)."""
        return self.total_rwa * 0.08
    
    def get_breakdown(self, by: str) -> Dict[str, Any]:
        """
        Get RWA breakdown by a specific field.
        
        Args:
            by: Field to get breakdown for (e.g., 'rating', 'segment', 'date')
                Should correspond to a key in the summary dictionary like 'by_rating'
                
        Returns:
            Dictionary with breakdown data or empty dict if not available
            
        Examples:
            >>> result.get_breakdown('rating')  # Returns breakdown by rating
            >>> result.get_breakdown('date')    # Returns breakdown by date
        """
        key = f'by_{by}'
        
        # Support both new format (by_rating) and old format (rwa_by_rating)
        if key in self.summary:
            return self.summary[key]
        elif f'rwa_by_{by}' in self.summary:
            return self.summary[f'rwa_by_{by}']
        return {}
    
    def has_breakdown(self, by: str) -> bool:
        """Check if breakdown by a specific field is available."""
        key = f'by_{by}'
        old_key = f'rwa_by_{by}'
        return key in self.summary or old_key in self.summary
    
    def get_available_breakdowns(self) -> List[str]:
        """Get list of all available breakdown types."""
        breakdowns = []
        
        # Look for both new format (by_*) and old format (rwa_by_*)
        for key in self.summary.keys():
            if key.startswith('by_'):
                breakdowns.append(key[3:])  # Remove 'by_' prefix
            elif key.startswith('rwa_by_'):
                breakdowns.append(key[7:])  # Remove 'rwa_by_' prefix
                
        return breakdowns
    
    def __str__(self) -> str:
        """String representation of the result."""
        result = (
            f"RWA Calculation Result:\n"
            f"  Total Exposure: {self.total_exposure:,.2f}\n"
            f"  Total RWA: {self.total_rwa:,.2f}\n"
            f"  Capital Requirement: {self.capital_requirement:,.2f}\n"
            f"  Average Risk Weight: {self.summary['average_risk_weight']:.2%}"
        )
        
        # Add information about available breakdowns
        breakdowns = self.get_available_breakdowns()
        if breakdowns:
            result += f"\n  Available Breakdowns: {', '.join(breakdowns)}"
            
        return result
