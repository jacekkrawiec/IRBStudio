# AIRB simulation and RWA logic

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, List, Optional, Union

from irbstudio.engine.base import BaseRWACalculator, RWAResult
from irbstudio.utils.logging import get_logger

class AIRBMortgageCalculator(BaseRWACalculator):
    """
    Advanced Internal Ratings-Based (AIRB) RWA calculator for mortgage exposures.
    
    Implements the Basel III AIRB formula for calculating risk weights and 
    risk-weighted assets (RWA) for residential mortgage exposures.
    """
    
    def __init__(self, regulatory_params: Dict[str, Any]):
        """
        Initialize the calculator with regulatory parameters.
        
        Args:
            regulatory_params: A dictionary containing regulatory parameters.
                Required keys:
                - asset_correlation: The asset correlation parameter (R).
                - confidence_level: The confidence level for the VaR calculation (default 0.999).
                - lgd: Loss Given Default (can be overridden at the exposure level).
                - maturity_adjustment: Whether to apply maturity adjustment (default False for mortgages).
        """
        super().__init__(regulatory_params)
        self.asset_correlation = regulatory_params.get('asset_correlation', 0.15)
        self.confidence_level = regulatory_params.get('confidence_level', 0.999)
        self.lgd = regulatory_params.get('lgd', 0.25)  # Default LGD if not provided per exposure
        self.maturity_adjustment = regulatory_params.get('maturity_adjustment', False)
        self.required_columns = ['exposure', 'pd']
        self.logger = get_logger(__name__)
        
    def calculate_rw(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk weights for each exposure in the portfolio.
        
        Implements the Basel AIRB RW formula:
        RW = LGD × N[(1-R)^(-0.5) × G(PD) + (R/(1-R))^(0.5) × G(0.999)]
        
        Args:
            portfolio_df: DataFrame containing the portfolio data.
                         Must include 'exposure', 'pd', and optionally 'lgd'.
                         
        Returns:
            DataFrame with original data plus a 'risk_weight' column.
        """
        self.validate_inputs(portfolio_df, self.required_columns)
        
        # Create a copy to avoid modifying the original
        result_df = portfolio_df.copy()
        
        # Ensure PD is between 0 and 1, and not exactly 0 or 1 (for numerical stability)
        # applies floor from CRR3 (0.05%)
        result_df['pd'] = result_df['pd'].clip(0.0005, 0.9999)
        
        # Use exposure-level LGD if available, otherwise use default
        if 'lgd' not in result_df.columns:
            result_df['lgd'] = self.lgd
        
        # AIRB RW formula components
        R = self.asset_correlation
        
        # Calculate risk weight using vectorized operations
        sqrt_1_minus_R = np.sqrt(1 - R)
        sqrt_R_div_1_minus_R = np.sqrt(R / (1 - R))
        
        # Normal inverse of PD
        norm_inverse_pd = norm.ppf(result_df['pd'])
        
        # Normal inverse of confidence level (usually 0.999)
        norm_inverse_conf = norm.ppf(self.confidence_level)
        
        # Core AIRB formula
        N_term = norm.cdf(
            (norm_inverse_pd / sqrt_1_minus_R) + 
            (sqrt_R_div_1_minus_R * norm_inverse_conf)
        )
        
        # Calculate risk weight as a percentage
        result_df['risk_weight'] = result_df['lgd'] * N_term
        
        # For regulatory reporting, multiply by 12.5 (reciprocal of 8% minimum capital)
        result_df['risk_weight'] = result_df['risk_weight'] * 12.5
        
        # If needed, apply maturity adjustment (uncommon for mortgages)
        if self.maturity_adjustment and 'maturity' in result_df.columns:
            # Simplified maturity adjustment formula
            b = (0.11852 - 0.05478 * np.log(result_df['pd'])) ** 2
            maturity_adj = (1 + (result_df['maturity'] - 2.5) * b) / (1 - 1.5 * b)
            result_df['risk_weight'] = result_df['risk_weight'] * maturity_adj
        
        return result_df
    
    def calculate_rwa(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RWA for each exposure in the portfolio.
        
        Args:
            portfolio_df: DataFrame containing the portfolio data.
                         
        Returns:
            DataFrame with original data plus 'risk_weight' and 'rwa' columns.
        """
        result_df = self.calculate_rw(portfolio_df)
        
        # Calculate RWA as exposure * risk_weight
        result_df['rwa'] = result_df['exposure'] * result_df['risk_weight']
        
        return result_df
    
    def calculate(self, portfolio_df: pd.DataFrame) -> RWAResult:
        """
        Calculate RWA and return a structured result.
        
        This is a convenience method that wraps calculate_rwa() and summarize_rwa().
        
        Args:
            portfolio_df: DataFrame containing the portfolio data.
            
        Returns:
            RWAResult object with portfolio, summary, and metadata.
        """
        self.logger.info("Starting AIRB RWA calculation for mortgage portfolio")
        
        # Calculate RWA
        result_df = self.calculate_rwa(portfolio_df)
        
        # Generate summary statistics
        summary = self.summarize_rwa(result_df)
        
        # Add metadata
        metadata = {
            'calculator_type': 'AIRB',
            'asset_class': 'Mortgage',
            'asset_correlation': self.asset_correlation,
            'confidence_level': self.confidence_level,
            'average_pd': result_df['pd'].mean(),
            'exposure_weighted_pd': (result_df['pd'] * result_df['exposure']).sum() / result_df['exposure'].sum(),
        }
        
        self.logger.info(f"AIRB RWA calculation completed. Total RWA: {summary['total_rwa']:,.2f}")
        
        return RWAResult(result_df, summary, metadata)
