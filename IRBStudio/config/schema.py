"""Pydantic models for configuration validation.

Defines:
- ColumnMapping: maps user column names to canonical internal names.
- RegulatoryParams: basic regulatory defaults used by calculators.
- Scenario: a user-defined scenario (name + pd_auc + optional overrides).
- Config: top-level configuration container.

This module is intentionally small for the MVP and can be extended.
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field, field_validator



class ColumnMapping(BaseModel):
	"""Mapping from canonical field name -> user column name.
	Example:
		column_mapping:
			loan_id: loan_identifier
			exposure: balance
			ltv: ltv
	"""

	loan_id: str = Field("loan_id", description="Unique loan identifier column name")
	exposure: str = Field("balance", description="Exposure at default / balance column name")
	ltv: Optional[str] = Field(None, description="Loan-to-value column name (optional)")
	term_remaining_months: Optional[str] = Field(
		None, description="Remaining term in months (optional)"
	)
	origination_date: Optional[str] = Field(None, description="Origination date column (optional)")

	@staticmethod
	def get_required_fields() -> List[str]:
		"""Returns the list of canonical field names that are required for a portfolio."""
		return ["loan_id", "exposure"]



class RegulatoryParams(BaseModel):
	"""Minimal set of regulatory parameters with sensible defaults.

	Defaults are placeholders and should be reviewed by domain experts.
	"""

	jurisdiction: str = Field("generic", description="Jurisdiction name for defaults")
	asset_correlation: float = Field(
		0.15, description="Asset correlation used in AIRB formula (placeholder default)"
	)
	confidence_level: float = Field(0.999, description="Confidence level for capital calculation")

	@field_validator("asset_correlation")
	def check_corr_range(cls, v):
		if not (0.0 <= v <= 1.0):
			raise ValueError("asset_correlation must be between 0 and 1")
		return v



class Scenario(BaseModel):
	"""User-defined scenario.

	For MVP, scenarios are driven by a desired PD model AUC.
	"""

	name: str
	pd_auc: float = Field(..., gt=0.0, lt=1.0, description="Target AUC for simulated PDs")
	description: Optional[str] = None
	overrides: Optional[Dict[str, float]] = None



class Config(BaseModel):
	"""Top-level configuration container for the analysis run."""

	column_mapping: ColumnMapping = Field(default_factory=ColumnMapping)
	regulatory: RegulatoryParams = Field(default_factory=RegulatoryParams)
	scenarios: List[Scenario] = Field(..., min_length=1)
