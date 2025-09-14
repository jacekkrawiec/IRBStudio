import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from irbstudio.simulation.risk_ranker import train_risk_ranker

@pytest.fixture
def sample_historical_data():
    """Fixture for a sample historical dataset."""
    data = {
        'exposure': [100, 200, 150, 50, 300, 400, 250, 120, 80, 180],
        'ltv': [0.8, 0.9, 0.85, 0.7, 0.95, 0.6, 0.75, 0.88, 0.92, 0.78],
        'default_flag': [1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)

def test_train_risk_ranker_success(sample_historical_data):
    """Test that the ranker trains successfully with valid data."""
    features = ['exposure', 'ltv']
    target = 'default_flag'
    
    model = train_risk_ranker(sample_historical_data, features, target)
    
    assert isinstance(model, LogisticRegression)
    # Check if the model is fitted
    assert hasattr(model, 'coef_')

def test_train_risk_ranker_predict_proba(sample_historical_data):
    """Test that the trained model can produce probability predictions."""
    features = ['exposure', 'ltv']
    target = 'default_flag'
    
    model = train_risk_ranker(sample_historical_data, features, target)
    
    # Create some dummy application data
    app_data = pd.DataFrame({
        'exposure': [110, 220],
        'ltv': [0.81, 0.89]
    })
    
    predictions = model.predict_proba(app_data[features])
    
    assert predictions.shape == (2, 2)
    assert np.all((predictions >= 0) & (predictions <= 1))

def test_train_risk_ranker_empty_data():
    """Test for ValueError when historical data is empty."""
    with pytest.raises(ValueError, match="Historical data cannot be empty"):
        train_risk_ranker(pd.DataFrame(), ['exposure'], 'default_flag')

def test_train_risk_ranker_missing_columns(sample_historical_data):
    """Test for ValueError when feature or target columns are missing."""
    features = ['exposure', 'non_existent_feature']
    target = 'default_flag'
    
    with pytest.raises(ValueError, match="Missing required columns"):
        train_risk_ranker(sample_historical_data, features, target)
        
    features = ['exposure', 'ltv']
    target = 'non_existent_target'
    
    with pytest.raises(ValueError, match="Missing required columns"):
        train_risk_ranker(sample_historical_data, features, target)

def test_train_risk_ranker_single_feature(sample_historical_data):
    """Test that the ranker trains successfully with a single feature."""
    features = ['exposure']
    target = 'default_flag'
    
    model = train_risk_ranker(sample_historical_data, features, target)
    
    assert isinstance(model, LogisticRegression)
    assert model.coef_.shape == (1, 1)
