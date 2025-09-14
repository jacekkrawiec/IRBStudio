import pytest
import numpy as np
from irbstudio.simulation.distribution import BetaMixtureFitter
from irbstudio.simulation.score_generation import (
    find_auc_calibration_factor,
    generate_calibrated_scores,
    _calculate_auc
)

@pytest.fixture
def fitted_dist():
    """A simple, fitted BetaMixtureFitter for testing."""
    np.random.seed(42)
    # Create a unimodal distribution for simplicity
    data = np.random.beta(a=5, b=20, size=2000)
    fitter = BetaMixtureFitter(n_components=1).fit(data)
    return fitter

def test_calculate_auc():
    """Test the internal AUC calculation helper."""
    # Perfect separation
    scores_good = np.array([0.1, 0.2, 0.3])
    scores_bad = np.array([0.4, 0.5, 0.6])
    assert _calculate_auc(scores_good, scores_bad) == 1.0

    # No separation
    scores_good = np.array([0.1, 0.5, 0.9])
    scores_bad = np.array([0.1, 0.5, 0.9])
    assert _calculate_auc(scores_good, scores_bad) == 0.5

    # Random scores
    np.random.seed(42)
    scores_good = np.random.rand(500)
    scores_bad = np.random.rand(500)
    # Should be around 0.5
    assert 0.45 < _calculate_auc(scores_good, scores_bad) < 0.55

def test_find_gamma_for_target_auc(fitted_dist):
    """Test if the optimization can find a gamma for a reasonable target AUC."""
    target_auc = 0.75
    gamma = find_auc_calibration_factor(fitted_dist, target_auc, n_samples_per_dist=20000)
    
    assert gamma > 1.0  # Gamma must be > 1 to create separation

    # Verify that the found gamma produces an AUC close to the target
    scores_good, scores_bad = generate_calibrated_scores(
        fitted_dist, gamma, n_good=20000, n_bad=20000
    )
    actual_auc = _calculate_auc(scores_good, scores_bad)
    
    # Use a slightly larger tolerance due to stochastic nature of sampling
    assert actual_auc == pytest.approx(target_auc, abs=0.015)

def test_gamma_for_auc_half_is_one(fitted_dist):
    """Test that for a target AUC of 0.5, gamma is 1."""
    gamma = find_auc_calibration_factor(fitted_dist, target_auc=0.5)
    assert gamma == 1.0

def test_unachievable_auc_raises_error(fitted_dist):
    """Test that an unachievably high AUC raises a ValueError."""
    # This AUC is likely impossible to achieve with this method and tight bounds
    with pytest.raises(ValueError, match="Target AUC of 0.999 is not achievable"):
        find_auc_calibration_factor(fitted_dist, target_auc=0.999, gamma_bounds=(1, 1.5))

def test_invalid_auc_raises_error(fitted_dist):
    """Test that an AUC outside the [0.5, 1.0) range raises an error."""
    with pytest.raises(ValueError, match="Target AUC must be between 0.5 and 1.0"):
        find_auc_calibration_factor(fitted_dist, target_auc=0.4)
    with pytest.raises(ValueError, match="Target AUC must be between 0.5 and 1.0"):
        find_auc_calibration_factor(fitted_dist, target_auc=1.0)

def test_generate_calibrated_scores(fitted_dist):
    """Test the score generation function."""
    gamma = 2.0
    n_good, n_bad = 100, 50
    
    scores_good, scores_bad = generate_calibrated_scores(
        fitted_dist, gamma, n_good=n_good, n_bad=n_bad
    )

    assert len(scores_good) == n_good
    assert len(scores_bad) == n_bad
    
    # With gamma > 1, bad scores should be higher on average than good scores
    # because the transformation s^(1/gamma) increases scores, while s^gamma
    # decreases them (for s < 1).
    assert np.mean(scores_bad) > np.mean(scores_good)
