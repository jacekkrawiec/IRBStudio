import pytest
import numpy as np
from irbstudio.simulation.distribution import BetaMixtureFitter

@pytest.fixture
def sample_bimodal_data():
    """Generate data from a known Beta mixture to test parameter recovery."""
    np.random.seed(42)
    # Component 1: Low PDs, high concentration
    data1 = np.random.beta(a=2, b=50, size=300)
    # Component 2: Higher PDs, more spread out
    data2 = np.random.beta(a=10, b=30, size=700)
    return np.concatenate([data1, data2])

def test_fitter_initialization():
    """Test that the fitter initializes correctly."""
    fitter = BetaMixtureFitter(n_components=3, tol=1e-5, max_iter=200)
    assert fitter.n_components == 3
    assert fitter.tol == 1e-5
    assert fitter.max_iter == 200
    assert fitter.weights_ is None

def test_fitter_runs_without_error(sample_bimodal_data):
    """Test that the fit method runs to completion without raising errors."""
    fitter = BetaMixtureFitter(n_components=2)
    try:
        fitter.fit(sample_bimodal_data)
    except Exception as e:
        pytest.fail(f"BetaMixtureFitter.fit() raised an exception: {e}")
    
    assert fitter.weights_ is not None
    assert fitter.alphas_ is not None
    assert fitter.betas_ is not None
    assert len(fitter.weights_) == 2
    assert len(fitter.alphas_) == 2
    assert len(fitter.betas_) == 2

def test_parameter_recovery(sample_bimodal_data):
    """
    Test if the fitter can reasonably recover the parameters of a known mixture.
    This is a sanity check, as exact recovery is not expected.
    """
    fitter = BetaMixtureFitter(n_components=2, max_iter=150)
    fitter.fit(sample_bimodal_data)

    # Known parameters (weights are approx 0.3 and 0.7)
    known_weights = np.array([0.3, 0.7])
    known_alphas = np.array([2, 10])
    known_betas = np.array([50, 30])

    # Sort the fitted parameters by weight to compare with known params
    sort_indices = np.argsort(fitter.weights_)
    fitted_weights = fitter.weights_[sort_indices]
    fitted_alphas = fitter.alphas_[sort_indices]
    fitted_betas = fitter.betas_[sort_indices]

    # Check if weights are close
    np.testing.assert_allclose(fitted_weights, known_weights, atol=0.1)
    
    # Check if the component with lower weight has alpha close to 2
    np.testing.assert_allclose(fitted_alphas[0], known_alphas[0], atol=1.5)
    np.testing.assert_allclose(fitted_betas[0], known_betas[0], atol=15)
    
    # Check if the component with higher weight has alpha close to 10
    np.testing.assert_allclose(fitted_alphas[1], known_alphas[1], atol=3)
    np.testing.assert_allclose(fitted_betas[1], known_betas[1], atol=10)

def test_predict_proba(sample_bimodal_data):
    """Test the predict_proba method."""
    fitter = BetaMixtureFitter(n_components=2).fit(sample_bimodal_data)
    probs = fitter.predict_proba(sample_bimodal_data)
    
    assert probs.shape == (len(sample_bimodal_data), 2)
    # Probabilities for each data point should sum to 1
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-6)

def test_sample_method(sample_bimodal_data):
    """Test the sample method."""
    fitter = BetaMixtureFitter(n_components=2).fit(sample_bimodal_data)
    n_samples = 500
    samples = fitter.sample(n_samples)
    
    assert samples.shape == (n_samples,)
    # All sampled values should be between 0 and 1
    assert np.all((samples >= 0) & (samples <= 1))
    # The mean of the samples should be reasonably close to the mean of the original data
    assert np.mean(samples) == pytest.approx(np.mean(sample_bimodal_data), abs=0.05)

def test_unfitted_model_raises_error(sample_bimodal_data):
    """Test that calling predict or sample on an unfitted model raises an error."""
    fitter = BetaMixtureFitter(n_components=2)
    with pytest.raises(RuntimeError):
        fitter.predict_proba(sample_bimodal_data)
    with pytest.raises(RuntimeError):
        fitter.sample(10)
