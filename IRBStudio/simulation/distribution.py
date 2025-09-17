"""
This module provides tools for fitting mixture distributions, specifically
the Beta Mixture Model, which is essential for modeling PD scores.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from ..utils.logging import get_logger
import numpy as np

logger = get_logger(__name__)

class BetaMixtureFitter:
    """
    Fits a Beta Mixture Model to 1-dimensional data.

    This class supports two modes:
    1. Unsupervised: Fits a mixture using the Expectation-Maximization (EM) algorithm
       if only `X` is provided to `fit`.
    2. Supervised: Fits two separate Beta distributions if both `X` and `y` (labels)
       are provided. This is the preferred method for modeling default vs. non-default scores.

    Attributes:
        n_components (int): The number of Beta distributions in the mixture.
        tol (float): The convergence tolerance for the log-likelihood (unsupervised mode).
        max_iter (int): The maximum number of EM iterations (unsupervised mode).
        weights_ (np.ndarray): The mixing weights for each component.
        alphas_ (np.ndarray): The 'alpha' parameters for each component's Beta distribution.
        betas_ (np.ndarray): The 'beta' parameters for each component's Beta distribution.
    """

    def __init__(self, n_components: int = 2, tol: float = 1e-4, max_iter: int = 100):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.weights_ = None
        self.alphas_ = None
        self.betas_ = None

    def _initialize_params(self, X: np.ndarray):
        """
        Initializes the model parameters using KMeans (for unsupervised fitting).
        """
        kmeans = KMeans(n_clusters=self.n_components, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X.reshape(-1, 1))

        self.weights_ = np.zeros(self.n_components)
        self.alphas_ = np.zeros(self.n_components)
        self.betas_ = np.zeros(self.n_components)

        for i in range(self.n_components):
            data_component = X[labels == i]
            if len(data_component) == 0:
                self.weights_[i] = 1 / self.n_components
                self.alphas_[i] = 1
                self.betas_[i] = 1
                continue

            self.weights_[i] = len(data_component) / len(X)
            a, b, _, _ = beta.fit(data_component, floc=0, fscale=1)
            self.alphas_[i] = a
            self.betas_[i] = b

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the Expectation (E) step of the EM algorithm.
        """
        responsibilities = np.zeros((len(X), self.n_components))
        for i in range(self.n_components):
            responsibilities[:, i] = self.weights_[i] * beta.pdf(X, self.alphas_[i], self.betas_[i])
        
        sum_resp = np.sum(responsibilities, axis=1)[:, np.newaxis]
        responsibilities /= np.where(sum_resp == 0, 1, sum_resp)
        return responsibilities

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        Performs the Maximization (M) step of the EM algorithm.
        """
        self.weights_ = np.mean(responsibilities, axis=0)

        for i in range(self.n_components):
            def neg_log_likelihood(params):
                a, b = params
                if a <= 0 or b <= 0:
                    return np.inf
                return -np.sum(responsibilities[:, i] * beta.logpdf(X, a, b))

            initial_guess = [self.alphas_[i], self.betas_[i]]
            result = minimize(neg_log_likelihood, initial_guess, bounds=((1e-6, None), (1e-6, None)))
            
            self.alphas_[i], self.betas_[i] = result.x

    def _log_likelihood(self, X: np.ndarray) -> float:
        """Calculates the total log-likelihood of the data given the model."""
        likelihoods = np.zeros((len(X), self.n_components))
        for i in range(self.n_components):
            likelihoods[:, i] = self.weights_[i] * beta.pdf(X, self.alphas_[i], self.betas_[i])
        return np.sum(np.log(np.sum(likelihoods, axis=1)))

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fits the Beta Mixture Model to the data.

        If `y` is provided, it performs supervised fitting.
        If `y` is None, it performs unsupervised fitting using EM.

        Args:
            X (np.ndarray): A 1D numpy array of data, with values between 0 and 1.
            y (np.ndarray, optional): A 1D numpy array of binary labels (0 or 1).
                                      If provided, `n_components` must be 2.
        """
        X = np.clip(X, 1e-6, 1 - 1e-6)

        if y is not None:
            self.X_train_ = X
            self.y_train_ = y
            self._fit_supervised(X, y)
        else:
            self._fit_unsupervised(X)
            
        return self

    def _fit_supervised(self, X: np.ndarray, y: np.ndarray):
        """
        Fits two separate Beta distributions to the data based on the binary label y.
        Component 0: Non-Default (y=0)
        Component 1: Default (y=1)
        """
        if self.n_components != 2:
            raise ValueError("Supervised fitting requires n_components=2.")

        X_non_default = X[y == 0]
        X_default = X[y == 1]

        if len(X_non_default) < 2:
            raise ValueError("Not enough data for the non-defaulting class (y=0) to fit a distribution.")
        if len(X_default) < 2:
            raise ValueError("Not enough data for the defaulting class (y=1) to fit a distribution.")

        # Fit beta for non-defaulting (Component 0)
        alpha_nd, beta_nd, _, _ = beta.fit(X_non_default, floc=0, fscale=1)

        # Fit beta for defaulting (Component 1)
        alpha_d, beta_d, _, _ = beta.fit(X_default, floc=0, fscale=1)

        self.alphas_ = np.array([alpha_nd, alpha_d])
        self.betas_ = np.array([beta_nd, beta_d])
        
        weight_nd = len(X_non_default) / len(X)
        self.weights_ = np.array([weight_nd, 1 - weight_nd])
        
        logger.info("Supervised fitting complete.")
        logger.info(f"Non-Default (C0): alpha={alpha_nd:.2f}, beta={beta_nd:.2f}, weight={self.weights_[0]:.2f}")
        logger.info(f"Default (C1): alpha={alpha_d:.2f}, beta={beta_d:.2f}, weight={self.weights_[1]:.2f}")

    def _fit_unsupervised(self, X: np.ndarray):
        """
        Fits the Beta Mixture Model using the EM algorithm.
        """
        self._initialize_params(X)
        
        prev_log_likelihood = -np.inf
        
        for i in range(self.max_iter):
            try:
                responsibilities = self._e_step(X)
                self._m_step(X, responsibilities)
                
                current_log_likelihood = self._log_likelihood(X)
                
                if np.abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                    logger.info(f"Converged after {i+1} iterations.")
                    break
                
                prev_log_likelihood = current_log_likelihood
            except Exception as e:
                logger.error(f"Error during EM iteration {i}: {e}")
                break
        else:
            logger.warning(f"Did not converge after {self.max_iter} iterations.")
        
        logger.info("Unsupervised fitting complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of each data point belonging to each component.
        """
        if self.weights_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        X = np.clip(X, 1e-6, 1 - 1e-6)
        return self._e_step(X)

    def sample(self, n_samples: int, component: int = None, target_auc: float = None) -> np.ndarray:
        """
        Generates random samples from the fitted mixture model.
        
        Args:
            n_samples (int): Number of samples to generate.
            component (int, optional): If provided, samples only from the specified component (0-indexed).
            target_auc (float, optional): If provided, calibrates the samples to achieve this AUC
                                          using the training data. Requires supervised fitting.
        
        Returns:
            np.ndarray: Array of generated samples.
        """
        if self.weights_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        if component is not None:
            samples = beta.rvs(self.alphas_[component], self.betas_[component], size=n_samples)
        else:
            # Choose components based on weights
            component_choices = np.random.choice(self.n_components, size=n_samples, p=self.weights_)

            # Sample from each chosen component
            samples = np.zeros(n_samples)
            for i in range(self.n_components):
                idx = (component_choices == i)
                n_component_samples = np.sum(idx)
                if n_component_samples > 0:
                    samples[idx] = beta.rvs(self.alphas_[i], self.betas_[i], size=n_component_samples)
        
        # Apply AUC calibration if requested
        if target_auc is not None:
            if not hasattr(self, 'y_train_'):
                raise ValueError("Target AUC calibration requires supervised fitting with labels.")
            from score_generation import find_auc_calibration_factor
            gamma = find_auc_calibration_factor(self, target_auc)
            samples = (samples ** gamma) / ((samples ** gamma) + ((1 - samples) ** gamma))
            logger.info(f"Applied AUC calibration with gamma={gamma:.3f} for target AUC {target_auc:.3f}")
                
        return samples
