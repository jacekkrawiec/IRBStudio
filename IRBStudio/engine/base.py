# Abstract base classes for calculators

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


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
    def calculate_rwa(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RWA for each exposure in the portfolio.
        
        This typically calls calculate_rw() first and then multiplies by exposure.
        
        Args:
            portfolio_df: DataFrame containing the portfolio data.
                         Must include required fields specific to the calculator.
                         
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
    
    def summarize_rwa(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Provide a summary of RWA calculation results.
        
        Args:
            portfolio_df: DataFrame with calculated risk weights and RWA.
            
        Returns:
            Dictionary with summary statistics such as total RWA, average risk weight, etc.

        TODO: parametrize summary to include breakdown by user specified fields (e.g., rating, segment).
        TODO: make sure it summarizes by date if multiple dates are present -> the same applies to calculate method if needed.
        """
        if 'rwa' not in portfolio_df.columns:
            raise ValueError("RWA must be calculated before summarizing.")
        
        result = {
            'total_rwa': portfolio_df['rwa'].sum(),
            'average_risk_weight': portfolio_df['risk_weight'].mean(),
            'total_exposure': portfolio_df['exposure'].sum(),
            'weighted_average_rw': (portfolio_df['rwa'].sum() / portfolio_df['exposure'].sum())
        }
        
        # Add rating/segment breakdown if available
        if 'rating' in portfolio_df.columns:
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

    TODO: this class needs to be extended to support breakdowns by rating, segment, date, etc.
    """
    
    def __init__(self, 
                 portfolio_with_rwa: pd.DataFrame,
                 summary: Dict[str, Any],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize with calculation results.
        
        Args:
            portfolio_with_rwa: DataFrame with risk weights and RWA.
            summary: Dictionary with summary statistics.
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
    
    def __str__(self) -> str:
        """String representation of the result."""
        return (
            f"RWA Calculation Result:\n"
            f"  Total Exposure: {self.total_exposure:,.2f}\n"
            f"  Total RWA: {self.total_rwa:,.2f}\n"
            f"  Capital Requirement: {self.capital_requirement:,.2f}\n"
            f"  Average Risk Weight: {self.summary['average_risk_weight']:.2%}"
        )
