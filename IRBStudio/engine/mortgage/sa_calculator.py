# SA RWA calculation logic

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

from irbstudio.engine.base import BaseRWACalculator, RWAResult
from irbstudio.utils.logging import get_logger

class SAMortgageCalculator(BaseRWACalculator):
    """
    Standardized Approach (SA) RWA calculator for mortgage exposures.
    
    Implements the CRR3 (Capital Requirements Regulation) loan splitting approach for 
    calculating risk weights and risk-weighted assets (RWA) for residential mortgage exposures.
    
    For non-IPRA (income-producing real estate assets) residential exposures:
    - The portion of exposure up to 55% of the property value receives a 20% risk weight
    - The remaining portion is treated as unsecured and receives a 75% risk weight
    
    Reference: CRR3 Article 125 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1623

    TODO: Extend to cover IPRA and other property types (commercial, mixed-use, etc.)
    """
    
    def __init__(self, regulatory_params: Dict[str, Any]):
        """
        Initialize the calculator with regulatory parameters.
        
        Args:
            regulatory_params: A dictionary containing regulatory parameters.
                Optional keys:
                - secured_portion_rw: Risk weight for the secured portion (up to 55% of property value)
                - unsecured_portion_rw: Risk weight for the unsecured portion
                - property_value_threshold: Threshold for the secured portion (as % of property value)
        """
        super().__init__(regulatory_params)
        
        # CRR3 parameters for residential non-IPRA exposures
        self.secured_portion_rw = regulatory_params.get('secured_portion_rw', 0.20)  # 20% RW
        self.unsecured_portion_rw = regulatory_params.get('unsecured_portion_rw', 0.75)  # 75% RW
        self.property_value_threshold = regulatory_params.get('property_value_threshold', 0.55)  # 55% of property value
        
        self.required_columns = ['exposure', 'property_value']
        self.logger = get_logger(__name__)
    
    def calculate_rw(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk weights for each exposure in the portfolio using CRR3 loan splitting approach.
        
        According to CRR3 (Article 125), for residential exposures:
        - The portion of exposure up to 55% of the property value receives a 20% risk weight
        - The remaining portion is treated as unsecured and receives a 75% risk weight
        
        Args:
            portfolio_df: DataFrame containing the portfolio data.
                         Must include 'exposure' and 'property_value'.
                         
        Returns:
            DataFrame with original data plus 'risk_weight', 'secured_portion', 
            'unsecured_portion', 'secured_rwa', and 'unsecured_rwa' columns.
        """
        self.validate_inputs(portfolio_df, self.required_columns)
        
        # Create a copy to avoid modifying the original
        result_df = portfolio_df.copy()
        
        # Calculate threshold for the secured portion (55% of property value)
        result_df['secured_threshold'] = result_df['property_value'] * self.property_value_threshold
        
        # Split exposure into secured and unsecured portions
        result_df['secured_portion'] = result_df.apply(
            lambda row: min(row['exposure'], row['secured_threshold']), axis=1
        )
        result_df['unsecured_portion'] = result_df['exposure'] - result_df['secured_portion']
        
        # Calculate RWA for each portion
        result_df['secured_rwa'] = result_df['secured_portion'] * self.secured_portion_rw
        result_df['unsecured_rwa'] = result_df['unsecured_portion'] * self.unsecured_portion_rw
        
        # Calculate total RWA and effective risk weight
        result_df['rwa'] = result_df['secured_rwa'] + result_df['unsecured_rwa']
        result_df['risk_weight'] = result_df['rwa'] / result_df['exposure']
        
        # Handle potential division by zero
        result_df.loc[result_df['exposure'] == 0, 'risk_weight'] = 0
        
        self.logger.info("Applied CRR3 loan splitting approach for risk weight calculation")
        return result_df
    
    def calculate_rwa(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RWA for each exposure in the portfolio.
        
        Args:
            portfolio_df: DataFrame containing the portfolio data.
                         
        Returns:
            DataFrame with original data plus risk weight and RWA columns.
        """
        # The calculate_rw method now also calculates RWA
        result_df = self.calculate_rw(portfolio_df)
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
        self.logger.info("Starting SA RWA calculation for mortgage portfolio")
        
        # Calculate RWA
        result_df = self.calculate_rwa(portfolio_df)
        
        # Generate summary statistics
        summary = self.summarize_rwa(result_df)
        
        # Add metadata
        metadata = {
            'calculator_type': 'SA',
            'asset_class': 'Mortgage',
            'regulatory_framework': 'CRR3',
            'secured_portion_rw': self.secured_portion_rw,
            'unsecured_portion_rw': self.unsecured_portion_rw,
            'property_value_threshold': self.property_value_threshold,
            'avg_secured_portion_pct': (result_df['secured_portion'].sum() / result_df['exposure'].sum()) * 100,
            'avg_unsecured_portion_pct': (result_df['unsecured_portion'].sum() / result_df['exposure'].sum()) * 100
        }
        
        # Calculate LTV if possible
        if 'ltv' in result_df.columns:
            metadata.update({
                'average_ltv': result_df['ltv'].mean(),
                'exposure_weighted_ltv': (result_df['ltv'] * result_df['exposure']).sum() / result_df['exposure'].sum(),
            })
        elif 'property_value' in result_df.columns and result_df['property_value'].sum() > 0:
            # Calculate LTV from exposure and property value
            ltv_series = result_df['exposure'] / result_df['property_value']
            metadata.update({
                'average_ltv': ltv_series.mean(),
                'exposure_weighted_ltv': (ltv_series * result_df['exposure']).sum() / result_df['exposure'].sum(),
            })
        
        self.logger.info(f"SA RWA calculation completed. Total RWA: {summary['total_rwa']:,.2f}")
        
        return RWAResult(result_df, summary, metadata)
