"""
This module provides functions for generating calibrated credit scores
that are designed to meet a specific target Area Under the Curve (AUC).

The core idea is to take a base score distribution and apply a transformation
to create two new distributions—one for 'good' outcomes and one for 'bad'
outcomes—with a specified level of separation, as measured by AUC.
"""
import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
from .distribution import BetaMixtureFitter
from ..utils.logging import get_logger

logger = get_logger(__name__)

def _calculate_auc(scores_good: np.ndarray, scores_bad: np.ndarray) -> float:
    """
    A helper function to calculate the AUC given two arrays of scores.

    Args:
        scores_good (np.ndarray): Scores for the 'good' (non-default) class.
        scores_bad (np.ndarray): Scores for the 'bad' (default) class.

    Returns:
        float: The calculated Area Under the ROC Curve.
    """
    if len(scores_good) == 0 or len(scores_bad) == 0:
        return 0.5  # No basis for discrimination
    labels = np.concatenate([np.zeros(len(scores_good)), np.ones(len(scores_bad))])
    scores = np.concatenate([scores_good, scores_bad])
    return roc_auc_score(labels, scores)


def find_auc_calibration_factor(
    base_distribution: BetaMixtureFitter,
    target_auc: float,
    n_samples_per_dist: int = 10000,
    gamma_bounds: tuple = (1.0, 25.0),
    tolerance: float = 1e-4
) -> float:
    """
    Finds the calibration factor (gamma) needed to transform a base score
    distribution to achieve a target AUC between two derived distributions.

    The transformation is `s_good = s^(1/gamma)` and `s_bad = s^gamma`. A gamma of 1
    implies no transformation and results in an AUC of 0.5. As gamma increases,
    the separation between the two distributions grows, and the AUC increases.

    Args:
        base_distribution (BetaMixtureFitter): A fitted distribution model to sample from.
        target_auc (float): The desired AUC, between 0.5 and 1.0.
        n_samples_per_dist (int): The number of samples to draw for the optimization.
        gamma_bounds (tuple): The lower and upper bounds for the gamma search.
        tolerance (float): The convergence tolerance for the optimization.

    Returns:
        float: The calibrated gamma factor.
    """
    if not (0.5 <= target_auc < 1.0):
        raise ValueError("Target AUC must be between 0.5 and 1.0 (exclusive of 1.0).")
    if target_auc == 0.5:
        return 1.0

    # Define the objective function for the root-finding algorithm
    def objective(gamma: float):
        if gamma == 1.0:
            return 0.5 - target_auc

        # Generate scores from the base distribution
        base_scores = base_distribution.sample(n_samples_per_dist)
        
        # Transform scores to create separation. Clip to avoid issues with 0 or 1.
        # For AUC > 0.5, the 'bad' class (label 1) needs higher scores.
        # s^(1/gamma) pushes scores towards 1, s^gamma pushes them towards 0.
        base_scores = np.clip(base_scores, 1e-9, 1 - 1e-9)
        scores_bad = base_scores ** (1 / gamma)
        scores_good = base_scores ** gamma

        current_auc = _calculate_auc(scores_good, scores_bad)
        return current_auc - target_auc

    try:
        logger.info(f"Searching for gamma to achieve target AUC of {target_auc}...")
        calibrated_gamma, result = brentq(
            objective,
            a=gamma_bounds[0],
            b=gamma_bounds[1],
            xtol=tolerance,
            full_output=True
        )
        if not result.converged:
            logger.warning(f"Optimizer did not converge: {result.flag}")
            return result.root
        
        logger.info(f"Found calibrated gamma of {calibrated_gamma:.4f}.")
        return calibrated_gamma
    except ValueError:
        # This occurs if the objective function has the same sign at both bounds,
        # meaning the target AUC is likely unachievable.
        max_auc = objective(gamma_bounds[1]) + target_auc
        raise ValueError(
            f"Target AUC of {target_auc} is not achievable within the gamma bounds "
            f"{gamma_bounds}. The maximum achievable AUC for this distribution "
            f"and bounds is approximately {max_auc:.4f}."
        )


def generate_calibrated_scores(
    base_distribution: BetaMixtureFitter,
    gamma: float,
    n_good: int,
    n_bad: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates scores for 'good' and 'bad' populations using a calibrated
    transformation factor (gamma).

    Args:
        base_distribution (BetaMixtureFitter): A fitted distribution to sample from.
        gamma (float): The calibration factor from `find_auc_calibration_factor`.
        n_good (int): The number of 'good' scores to generate.
        n_bad (int): The number of 'bad' scores to generate.

    Returns:
        A tuple containing (scores_good, scores_bad).
    """
    # Generate scores for the good population
    base_scores_good = base_distribution.sample(n_good)
    scores_good = np.clip(base_scores_good, 1e-9, 1 - 1e-9) ** gamma

    # Generate scores for the bad population
    base_scores_bad = base_distribution.sample(n_bad)
    scores_bad = np.clip(base_scores_bad, 1e-9, 1 - 1e-9) ** (1 / gamma)
    
    return scores_good, scores_bad
