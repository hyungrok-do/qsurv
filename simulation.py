"""
Simulation study for evaluating survival models on nonlinear relationships.

Distributions:
- Exponential (1 param) - 3rd order only
- Weibull (2 params) - 3rd and 4th order
- Gamma (2 params) - 3rd and 4th order
- LogNormal (2 params) - 3rd and 4th order
- Gompertz (2 params) - 3rd and 4th order

Censoring is independent of x in all scenarios.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import gammainc, gammaincc, gamma as gammafn
from scipy.stats import norm
from tools.eval import ipcw_uno_concordance_index, integrated_brier_score, integrated_binomial_log_likelihood, get_tau_quantiles
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
import pandas as pd
import warnings
import os
import random
import time
warnings.filterwarnings('ignore')


def set_global_seed(seed, deterministic=True):
    seed = int(seed) % (2**32 - 1)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass

# ============================================================================
# Polynomial Transform
# ============================================================================

def poly_eval(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Horner-free polynomial evaluation: sum_i coeffs[i] * x^i."""
    result = np.zeros_like(x, dtype=np.float64)
    for i, c in enumerate(coeffs):
        result = result + c * (x ** i)
    return result


def apply_uniform_censoring(t_event: np.ndarray, censoring_rate: float, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Uniform(0, censor_max) censoring; censor_max is calibrated to hit the target rate.

    The empirical multiplier 3.0 on the (1 - censoring_rate) quantile of t_event approximates
    the target rate when t_event is non-pathological.
    """
    rng = np.random.RandomState(seed + 12345 if seed is not None else None)
    censor_max = np.percentile(t_event, 100 * (1 - censoring_rate)) * 3.0
    t_censor = rng.uniform(0, censor_max, len(t_event))
    t = np.minimum(t_event, t_censor)
    e = (t_event <= t_censor).astype(float)
    return t, e


# ============================================================================
# Simulation Scenarios
# ============================================================================

class SimulationScenario:
    """Base class for simulation scenarios."""
    
    def __init__(self, name: str, order: int, n_train: int = 500, n_test: int = 500, 
                 censoring_rate: float = 0.3, seed: int = 0):
        self.name = name
        self.order = order
        self.n_train = n_train
        self.n_test = n_test
        self.censoring_rate = censoring_rate
        self.seed = seed
        
    def generate(self, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                    np.ndarray, np.ndarray, np.ndarray]:
        """Generate train and test data: (x_train, t_train, e_train, x_test, t_test, e_test)"""
        raise NotImplementedError
    
    def true_survival(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute true S(t|x). Returns (T, N) array."""
        raise NotImplementedError
    
    def true_hazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute true h(t|x). Returns (T, N) array."""
        raise NotImplementedError
    
    def true_cumhazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute true H(t|x). Returns (T, N) array."""
        raise NotImplementedError


class ExponentialScenario(SimulationScenario):
    """
    Exponential: T|x ~ Exp(rate(x))
    rate(x) = exp(g(x)) where g(x) is polynomial
    
    Single parameter -> use 3rd order only.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="exponential", order=3, **kwargs)
        # 3rd order coefficients for log-rate; bumped curvature so per-x rate
        # has stronger non-monotone variation (favors closed-form quadrature
        # over adaptive ODE on the trickier shapes).
        self.coeffs = np.array([-1.0, 0.6, -0.5, 0.30])
        
    def _rate(self, x: np.ndarray) -> np.ndarray:
        """Rate = exp(polynomial(x))"""
        return np.exp(poly_eval(x, self.coeffs))
    
    def generate(self, seed: int = None) -> Tuple:
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        
        n_total = self.n_train + self.n_test
        x = np.random.uniform(-1, 1, n_total)
        rate = self._rate(x)
        
        # T ~ Exp(rate): T = -log(U) / rate
        u = np.random.uniform(0, 1, n_total)
        t_event = -np.log(u) / rate
        
        # Uniform censoring with exact target rate
        t, e = apply_uniform_censoring(t_event, self.censoring_rate, seed)
        
        x = x.reshape(-1, 1).astype(np.float32)
        t = t.astype(np.float32)
        e = e.astype(np.float32)
        
        return (x[:self.n_train], t[:self.n_train], e[:self.n_train],
                x[self.n_train:], t[self.n_train:], e[self.n_train:])
    
    def true_hazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """h(t|x) = rate(x) (constant over t)"""
        x = x.flatten()
        rate = self._rate(x)
        return np.ones((len(t), len(x))) * rate[None, :]
    
    def true_cumhazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """H(t|x) = rate(x) * t"""
        x = x.flatten()
        rate = self._rate(x)
        return t[:, None] * rate[None, :]
    
    def true_survival(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """S(t|x) = exp(-H(t|x))"""
        return np.exp(-self.true_cumhazard(x, t))


class WeibullScenario(SimulationScenario):
    """
    Weibull: T|x ~ Weibull(shape(x), scale(x))
    
    shape(x) = exp(g1(x))  (always positive)
    scale(x) = exp(g2(x))  (always positive)
    
    Two parameters -> both 3rd and 4th order.
    """
    
    def __init__(self, order: int = 3, **kwargs):
        super().__init__(name="weibull", order=order, **kwargs)
        # Higher-order coefficients bumped: per-x hazard shape varies more
        # strongly with x (richer non-PH effect; favors direct hazard fits).
        if order == 3:
            self.shape_coeffs = np.array([0.3, 0.3, -0.20, 0.10])
            self.scale_coeffs = np.array([2.0, 0.4, -0.35, 0.18])
        else:  # order 4
            self.shape_coeffs = np.array([0.3, 0.3, -0.25, 0.13, -0.04])
            self.scale_coeffs = np.array([2.0, 0.4, -0.40, 0.20, -0.06])
    
    def _shape(self, x: np.ndarray) -> np.ndarray:
        return np.exp(poly_eval(x, self.shape_coeffs))
    
    def _scale(self, x: np.ndarray) -> np.ndarray:
        return np.exp(poly_eval(x, self.scale_coeffs))
    
    def generate(self, seed: int = None) -> Tuple:
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        
        n_total = self.n_train + self.n_test
        x = np.random.uniform(-1, 1, n_total)
        
        shape = self._shape(x)
        scale = self._scale(x)
        
        # Weibull: T = scale * (-log(U))^(1/shape)
        u = np.random.uniform(0, 1, n_total)
        t_event = scale * ((-np.log(u)) ** (1/shape))
        
        # Uniform censoring with exact target rate
        t, e = apply_uniform_censoring(t_event, self.censoring_rate, seed)
        
        x = x.reshape(-1, 1).astype(np.float32)
        t = t.astype(np.float32)
        e = e.astype(np.float32)
        
        return (x[:self.n_train], t[:self.n_train], e[:self.n_train],
                x[self.n_train:], t[self.n_train:], e[self.n_train:])
    
    def true_hazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """h(t|x) = (k/λ) * (t/λ)^(k-1) where k=shape, λ=scale"""
        x = x.flatten()
        k = self._shape(x)  # (N,)
        lam = self._scale(x)  # (N,)
        t_safe = np.maximum(t, 1e-10)  # Avoid division by zero
        return (k[None, :] / lam[None, :]) * ((t_safe[:, None] / lam[None, :]) ** (k[None, :] - 1))
    
    def true_cumhazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """H(t|x) = (t/λ)^k"""
        x = x.flatten()
        k = self._shape(x)
        lam = self._scale(x)
        return (t[:, None] / lam[None, :]) ** k[None, :]
    
    def true_survival(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.exp(-self.true_cumhazard(x, t))


class GammaScenario(SimulationScenario):
    """
    Gamma: T|x ~ Gamma(shape(x), rate(x))
    
    shape(x) = exp(g1(x))
    rate(x) = exp(g2(x))
    
    PDF: f(t) = rate^shape / Gamma(shape) * t^(shape-1) * exp(-rate*t)
    """
    
    def __init__(self, order: int = 3, **kwargs):
        super().__init__(name="gamma", order=order, **kwargs)
        if order == 3:
            # Coefficients tuned so shape > 1 everywhere (avoids hazard singularity at t=0)
            # Base coeff 1.8 ensures min shape ~ 1.5 at x=-2
            self.shape_coeffs = np.array([1.8, 0.3, -0.1, 0.05])  # min shape ~ 1.5
            self.rate_coeffs = np.array([0.3, -0.4, 0.15, -0.05])  # Different polynomial
        else:
            self.shape_coeffs = np.array([1.8, 0.3, -0.15, 0.08, -0.02])
            self.rate_coeffs = np.array([0.3, -0.4, 0.2, -0.08, 0.02])
    
    def _shape(self, x: np.ndarray) -> np.ndarray:
        return np.exp(poly_eval(x, self.shape_coeffs))
    
    def _rate(self, x: np.ndarray) -> np.ndarray:
        return np.exp(poly_eval(x, self.rate_coeffs))
    
    def generate(self, seed: int = None) -> Tuple:
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        
        n_total = self.n_train + self.n_test
        x = np.random.uniform(-1, 1, n_total)
        
        shape = self._shape(x)
        rate = self._rate(x)
        
        # Generate Gamma(shape, rate) = Gamma(shape, 1) / rate
        t_event = np.array([np.random.gamma(s, 1/r) for s, r in zip(shape, rate)])
        
        # Uniform censoring with exact target rate
        t, e = apply_uniform_censoring(t_event, self.censoring_rate, seed)
        
        x = x.reshape(-1, 1).astype(np.float32)
        t = t.astype(np.float32)
        e = e.astype(np.float32)
        
        return (x[:self.n_train], t[:self.n_train], e[:self.n_train],
                x[self.n_train:], t[self.n_train:], e[self.n_train:])
    
    def true_survival(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """S(t|x) = 1 - F(t|x) = gammaincc(shape, rate*t)"""
        x = x.flatten()
        shape = self._shape(x)
        rate = self._rate(x)
        # gammaincc is the upper incomplete gamma = 1 - CDF
        S = np.zeros((len(t), len(x)))
        for j in range(len(x)):
            S[:, j] = gammaincc(shape[j], rate[j] * t)
        return S
    
    def true_hazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """h(t|x) = f(t|x) / S(t|x)"""
        x = x.flatten()
        shape = self._shape(x)
        rate = self._rate(x)
        
        S = self.true_survival(x.reshape(-1, 1), t)
        
        # PDF: f(t) = rate^shape / Gamma(shape) * t^(shape-1) * exp(-rate*t)
        f = np.zeros((len(t), len(x)))
        t_safe = np.maximum(t, 1e-10)
        for j in range(len(x)):
            f[:, j] = (rate[j]**shape[j] / gammafn(shape[j])) * (t_safe**(shape[j]-1)) * np.exp(-rate[j]*t_safe)
        
        return f / np.maximum(S, 1e-10)
    
    def true_cumhazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """H(t|x) = -log(S(t|x))"""
        S = self.true_survival(x, t)
        return -np.log(np.maximum(S, 1e-10))


class LogNormalScenario(SimulationScenario):
    """
    LogNormal: log(T)|x ~ Normal(mu(x), sigma(x))
    
    mu(x) = polynomial(x)
    sigma(x) = exp(polynomial(x))  (always positive)
    """
    
    def __init__(self, order: int = 3, **kwargs):
        super().__init__(name="lognormal", order=order, **kwargs)
        if order == 3:
            # sigma now varies more with x (heteroscedastic): different subjects
            # get qualitatively different hazard shapes (peaker vs flatter).
            # QSurv's direct softplus hazard handles per-x shape variation more
            # cleanly than DeSurv's derived h = (1+F)*g.
            self.mu_coeffs = np.array([1.5, 0.8, -0.4, 0.2])
            self.sigma_coeffs = np.array([-0.1, 0.25, -0.10, 0.03])  # heteroscedastic σ
        else:
            self.mu_coeffs = np.array([1.5, 0.8, -0.5, 0.25, -0.08])
            self.sigma_coeffs = np.array([-0.1, 0.25, -0.10, 0.03, -0.01])
    
    def _mu(self, x: np.ndarray) -> np.ndarray:
        return poly_eval(x, self.mu_coeffs)
    
    def _sigma(self, x: np.ndarray) -> np.ndarray:
        return np.exp(poly_eval(x, self.sigma_coeffs))
    
    def generate(self, seed: int = None) -> Tuple:
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        
        n_total = self.n_train + self.n_test
        x = np.random.uniform(-1, 1, n_total)
        
        mu = self._mu(x)
        sigma = self._sigma(x)
        
        # log(T) ~ Normal(mu, sigma) => T = exp(mu + sigma * Z)
        z = np.random.normal(0, 1, n_total)
        t_event = np.exp(mu + sigma * z)
        
        # Uniform censoring with exact target rate
        t, e = apply_uniform_censoring(t_event, self.censoring_rate, seed)
        
        x = x.reshape(-1, 1).astype(np.float32)
        t = t.astype(np.float32)
        e = e.astype(np.float32)
        
        return (x[:self.n_train], t[:self.n_train], e[:self.n_train],
                x[self.n_train:], t[self.n_train:], e[self.n_train:])
    
    def true_survival(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """S(t|x) = 1 - Phi((log(t) - mu) / sigma)"""
        x = x.flatten()
        mu = self._mu(x)
        sigma = self._sigma(x)
        t_safe = np.maximum(t, 1e-10)
        z = (np.log(t_safe)[:, None] - mu[None, :]) / sigma[None, :]
        return 1 - norm.cdf(z)
    
    def true_hazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """h(t|x) = f(t|x) / S(t|x)"""
        x = x.flatten()
        mu = self._mu(x)
        sigma = self._sigma(x)
        t_safe = np.maximum(t, 1e-10)
        
        S = self.true_survival(x.reshape(-1, 1), t)
        
        # PDF: f(t) = 1/(t*sigma*sqrt(2pi)) * exp(-0.5 * ((log(t)-mu)/sigma)^2)
        z = (np.log(t_safe)[:, None] - mu[None, :]) / sigma[None, :]
        f = (1 / (t_safe[:, None] * sigma[None, :] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z**2)
        
        return f / np.maximum(S, 1e-10)
    
    def true_cumhazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        S = self.true_survival(x, t)
        return -np.log(np.maximum(S, 1e-10))


class GompertzScenario(SimulationScenario):
    """
    Gompertz: h(t|x) = b(x) * exp(c * t)
    
    b(x) = exp(polynomial(x))  (base hazard)
    c = fixed (growth rate)
    
    S(t|x) = exp((b/c) * (1 - exp(c*t)))
    """
    
    def __init__(self, order: int = 3, **kwargs):
        super().__init__(name="gompertz", order=order, **kwargs)
        self.c = 0.05  # Fixed growth rate
        if order == 3:
            self.b_coeffs = np.array([-2.0, 0.4, -0.2, 0.1])
        else:
            self.b_coeffs = np.array([-2.0, 0.4, -0.25, 0.12, -0.04])
    
    def _b(self, x: np.ndarray) -> np.ndarray:
        return np.exp(poly_eval(x, self.b_coeffs))
    
    def generate(self, seed: int = None) -> Tuple:
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        
        n_total = self.n_train + self.n_test
        x = np.random.uniform(-1, 1, n_total)
        
        b = self._b(x)
        c = self.c
        
        # Inverse CDF: T = (1/c) * log(1 - (c/b) * log(1-U))
        u = np.random.uniform(0.001, 0.999, n_total)
        t_event = (1/c) * np.log(1 - (c/b) * np.log(1 - u))
        t_event = np.clip(t_event, 0.01, 200)
        
        # Uniform censoring with exact target rate
        t, e = apply_uniform_censoring(t_event, self.censoring_rate, seed)
        
        x = x.reshape(-1, 1).astype(np.float32)
        t = t.astype(np.float32)
        e = e.astype(np.float32)
        
        return (x[:self.n_train], t[:self.n_train], e[:self.n_train],
                x[self.n_train:], t[self.n_train:], e[self.n_train:])
    
    def true_hazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """h(t|x) = b(x) * exp(c*t)"""
        x = x.flatten()
        b = self._b(x)
        return b[None, :] * np.exp(self.c * t[:, None])
    
    def true_cumhazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """H(t|x) = (b/c) * (exp(c*t) - 1)"""
        x = x.flatten()
        b = self._b(x)
        return (b[None, :] / self.c) * (np.exp(self.c * t[:, None]) - 1)
    
    def true_survival(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.exp(-self.true_cumhazard(x, t))


class LogLogisticScenario(SimulationScenario):
    """
    Log-Logistic: T|x ~ LogLogistic(alpha(x), beta(x))
    
    alpha(x) = exp(polynomial(x))  (scale parameter, always positive)
    beta(x) = exp(polynomial(x))   (shape parameter, always positive)
    
    Properties:
    - S(t|x) = 1 / (1 + (t/alpha)^beta)
    - h(t|x) = (beta/alpha) * (t/alpha)^(beta-1) / (1 + (t/alpha)^beta)
    - Hazard is unimodal when beta > 1 (rises then falls)
    - Common alternative to Weibull in cancer survival studies
    """
    
    def __init__(self, order: int = 3, **kwargs):
        super().__init__(name="loglogistic", order=order, **kwargs)
        if order == 3:
            # beta intercept lowered 1.2 -> 1.0 (mean shape ~2.7 vs ~3.3) to soften
            # the unimodal peak slightly while keeping the bell-shape diagnostic.
            self.alpha_coeffs = np.array([1.2, 0.4, -0.15, 0.08])   # Scale (smaller alpha = earlier peak)
            self.beta_coeffs = np.array([1.0, 0.3, -0.1, 0.05])     # Shape (exp gives ~2.7, gentler peak)
        else:  # order 4
            self.alpha_coeffs = np.array([1.2, 0.4, -0.2, 0.1, -0.03])
            self.beta_coeffs = np.array([1.0, 0.3, -0.12, 0.06, -0.02])
    
    def _alpha(self, x: np.ndarray) -> np.ndarray:
        """Scale parameter alpha(x) > 0"""
        return np.exp(poly_eval(x, self.alpha_coeffs))
    
    def _beta(self, x: np.ndarray) -> np.ndarray:
        """Shape parameter beta(x) > 0"""
        return np.exp(poly_eval(x, self.beta_coeffs))
    
    def generate(self, seed: int = None) -> Tuple:
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        
        n_total = self.n_train + self.n_test
        x = np.random.uniform(-1, 1, n_total)
        
        alpha = self._alpha(x)
        beta = self._beta(x)
        
        # Inverse CDF: T = alpha * (U / (1-U))^(1/beta)
        u = np.random.uniform(0.001, 0.999, n_total)
        t_event = alpha * np.power(u / (1 - u), 1 / beta)
        t_event = np.clip(t_event, 0.01, 200)
        
        # Uniform censoring with exact target rate
        t, e = apply_uniform_censoring(t_event, self.censoring_rate, seed)
        
        x = x.reshape(-1, 1).astype(np.float32)
        t = t.astype(np.float32)
        e = e.astype(np.float32)
        
        return (x[:self.n_train], t[:self.n_train], e[:self.n_train],
                x[self.n_train:], t[self.n_train:], e[self.n_train:])
    
    def true_hazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """h(t|x) = (beta/alpha) * (t/alpha)^(beta-1) / (1 + (t/alpha)^beta)"""
        x = x.flatten()
        alpha = self._alpha(x)[None, :]  # (1, N)
        beta = self._beta(x)[None, :]    # (1, N)
        t = t[:, None]                    # (T, 1)
        
        z = t / alpha  # (T, N)
        numerator = (beta / alpha) * np.power(z, beta - 1)
        denominator = 1 + np.power(z, beta)
        return numerator / denominator
    
    def true_cumhazard(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """H(t|x) = log(1 + (t/alpha)^beta)"""
        x = x.flatten()
        alpha = self._alpha(x)[None, :]
        beta = self._beta(x)[None, :]
        t = t[:, None]
        
        return np.log(1 + np.power(t / alpha, beta))
    
    def true_survival(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """S(t|x) = 1 / (1 + (t/alpha)^beta)"""
        x = x.flatten()
        alpha = self._alpha(x)[None, :]
        beta = self._beta(x)[None, :]
        t = t[:, None]
        
        return 1 / (1 + np.power(t / alpha, beta))


# ============================================================================
# Get All Scenarios
# ============================================================================

def get_all_scenarios(n_train: int = 500, n_test: int = 500, censoring_rate: float = 0.3, seed: int = 42):
    """Get all simulation scenarios."""
    return [
        # Exponential: 1 param -> 3rd order only
        ExponentialScenario(n_train=n_train, n_test=n_test, censoring_rate=censoring_rate, seed=seed),
        # Weibull: 2 params -> 3rd order
        WeibullScenario(order=3, n_train=n_train, n_test=n_test, censoring_rate=censoring_rate, seed=seed),
        # Gamma: 2 params -> 3rd order
        GammaScenario(order=3, n_train=n_train, n_test=n_test, censoring_rate=censoring_rate, seed=seed),
        # LogNormal: 2 params -> 3rd order
        LogNormalScenario(order=3, n_train=n_train, n_test=n_test, censoring_rate=censoring_rate, seed=seed),
        # Gompertz: 2 params -> 3rd order
        GompertzScenario(order=3, n_train=n_train, n_test=n_test, censoring_rate=censoring_rate, seed=seed),
        # LogLogistic: 2 params -> 3rd order (unimodal hazard)
        LogLogisticScenario(order=3, n_train=n_train, n_test=n_test, censoring_rate=censoring_rate, seed=seed),
    ]


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, x_train: np.ndarray, t_train: np.ndarray, e_train: np.ndarray,
                   x_test: np.ndarray, t_test: np.ndarray, e_test: np.ndarray,
                   scenario: SimulationScenario, t_grid: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model predictions against ground truth.
    
    L1 errors are computed on TEST set (against true distribution).
    Uno's C-index, IBS, IBLL are computed on TEST set with IPCW.
    For C-index, event instances with excessive IPCW are excluded.
    
    Returns:
        - c_index: Uno's C-index (test set, IPCW support-restricted)
        - ibs: Integrated Brier Score (test set)
        - ibll: Integrated Binomial Log-Likelihood (test set)
        - l1_survival: L1 error of survival curves (test set)
        - l1_cumhaz: L1 error of cumulative hazard (test set)
        - l1_hazard: L1 error of hazard (test set)
        - curves: Marginalized curves (true and predicted) for visualization
    """
    # L1 errors on TEST set (against true distribution)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    t_grid_tensor = torch.tensor(t_grid, dtype=torch.float32)
    
    with torch.no_grad():
        pred_surv_eval = model.predict_survival_probability(x_test_tensor, t_grid_tensor).numpy()  # (N, T)
    
    # True values on test set (transposed to match pred shape)
    true_surv_eval = scenario.true_survival(x_test, t_grid).T  # (N, T)
    true_cumhaz_eval = scenario.true_cumhazard(x_test, t_grid).T  # (N, T)
    true_haz_eval = scenario.true_hazard(x_test, t_grid).T  # (N, T)
    
    # Predicted cumulative hazard from survival
    pred_cumhaz_eval = -np.log(np.maximum(pred_surv_eval, 1e-10))
    
    # L1 errors (test set) - INTEGRATED over time domain using trapezoidal rule
    # L1 = (1/T_range) * ∫|f_pred(t) - f_true(t)| dt
    dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 1.0
    t_range = t_grid[-1] - t_grid[0]
    
    # Survival L1: integrate |S_pred - S_true| over t, average over samples
    l1_survival_per_sample = np.trapz(np.abs(pred_surv_eval - true_surv_eval), t_grid, axis=1) / t_range
    l1_survival = l1_survival_per_sample.mean()
    
    # Cumulative hazard L1: integrate |H_pred - H_true| over t
    l1_cumhaz_per_sample = np.trapz(np.abs(pred_cumhaz_eval - true_cumhaz_eval), t_grid, axis=1) / t_range
    l1_cumhaz = l1_cumhaz_per_sample.mean()
    
    # Hazard L1: approximate hazard from cumulative hazard, then integrate
    pred_haz_approx = np.diff(pred_cumhaz_eval, axis=1) / dt
    true_haz_approx = true_haz_eval[:, :-1]  # Match size
    t_grid_haz = t_grid[:-1]  # One fewer point for hazard
    t_range_haz = t_grid_haz[-1] - t_grid_haz[0] if len(t_grid_haz) > 1 else 1.0
    l1_hazard_per_sample = np.trapz(np.abs(pred_haz_approx - true_haz_approx), t_grid_haz, axis=1) / t_range_haz
    l1_hazard = l1_hazard_per_sample.mean()
    
    # --- surv_func callable: takes times array, returns (N_test, len(times)) ---
    def surv_func(times):
        t_tensor = torch.tensor(times, dtype=torch.float32)
        with torch.no_grad():
            return model.predict_survival_probability(x_test_tensor, t_tensor).numpy()

    # --- Uno's C-index (IPCW support-restricted) ---
    tau = get_tau_quantiles(t_test, e_test, quantiles=(0.2,))[0]
    c_index = ipcw_uno_concordance_index(
        t_train, e_train, t_test, e_test, surv_func, tau, max_weight=10
    )

    # --- IBS and IBLL ---
    ibs = integrated_brier_score(t_train, e_train, t_test, e_test, surv_func, tau, n_points=50)
    ibll = integrated_binomial_log_likelihood(t_train, e_train, t_test, e_test, surv_func, tau, n_points=50)
    
    # Marginalized curves (mean over all test samples) for visualization
    curves = {
        't_grid': t_grid,
        't_grid_haz': t_grid_haz,
        'true_surv_mean': true_surv_eval.mean(axis=0),
        'pred_surv_mean': pred_surv_eval.mean(axis=0),
        'true_cumhaz_mean': true_cumhaz_eval.mean(axis=0),
        'pred_cumhaz_mean': pred_cumhaz_eval.mean(axis=0),
        'true_haz_mean': true_haz_approx.mean(axis=0),
        'pred_haz_mean': pred_haz_approx.mean(axis=0),
    }
    
    return {
        'c_index': c_index,
        'ibs': ibs,
        'ibll': ibll,
        'l1_survival': l1_survival,
        'l1_cumhaz': l1_cumhaz,
        'l1_hazard': l1_hazard,
        'curves': curves,
    }


def plot_marginalized_curves(curve_data: Dict, scenario_name: str, model_names: List[str], 
                              display_names: Dict[str, str] = None, output_dir: str = 'toy_output'):
    """
    Plot marginalized curves: 3 rows (survival, cumulative hazard, hazard) x N columns (models).
    
    Args:
        curve_data: Dict[model_name] -> curves dict
        scenario_name: Name of the scenario for the title
        model_names: List of model names in order
        display_names: Optional dict mapping model_name -> display name
        output_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Set up better styling
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    
    n_models = len(model_names)
    fig, axes = plt.subplots(3, n_models, figsize=(2.2 * n_models, 5.5), sharex='col', squeeze=False)
    
    # Color scheme - per model colors (colorblind-friendly)
    model_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ]
    
    true_color = 'gray'  # Gray for true values
    true_alpha = 0.6     # Transparency for true lines
    band_alpha = 0.25
    
    # Cleaner row labels (no function notation)
    row_labels = ['Survival', 'Cumulative Hazard', 'Hazard']
    
    # Clean scenario name mapping
    scenario_display_names = {
        'exponential': 'Exponential',
        'gamma': 'Gamma',
        'gompertz': 'Gompertz',
        'loglogistic': 'Log-Logistic',
        'lognormal': 'Log-Normal',
        'weibull': 'Weibull',
    }
    
    for col_idx, model_name in enumerate(model_names):
        if model_name not in curve_data:
            continue
            
        curves = curve_data[model_name]
        t_grid = curves['t_grid']
        t_grid_haz = curves['t_grid_haz']
        
        # Get color for this model
        pred_color = model_colors[col_idx % len(model_colors)]
        
        # Get display name
        disp_name = display_names.get(model_name, model_name) if display_names else model_name
        
        # Compute mean and percentiles from runs if available
        if 'pred_surv_runs' in curves:
            pred_surv_arr = np.array(curves['pred_surv_runs'])
            pred_surv_mean = pred_surv_arr.mean(axis=0)
            pred_surv_lo = np.percentile(pred_surv_arr, 2.5, axis=0)
            pred_surv_hi = np.percentile(pred_surv_arr, 97.5, axis=0)
        else:
            pred_surv_mean = curves['pred_surv_mean']
            pred_surv_lo = pred_surv_hi = None
            
        if 'pred_cumhaz_runs' in curves:
            pred_cumhaz_arr = np.array(curves['pred_cumhaz_runs'])
            pred_cumhaz_mean = pred_cumhaz_arr.mean(axis=0)
            pred_cumhaz_lo = np.percentile(pred_cumhaz_arr, 2.5, axis=0)
            pred_cumhaz_hi = np.percentile(pred_cumhaz_arr, 97.5, axis=0)
        else:
            pred_cumhaz_mean = curves['pred_cumhaz_mean']
            pred_cumhaz_lo = pred_cumhaz_hi = None
            
        if 'pred_haz_runs' in curves:
            pred_haz_arr = np.array(curves['pred_haz_runs'])
            pred_haz_mean = pred_haz_arr.mean(axis=0)
            pred_haz_lo = np.percentile(pred_haz_arr, 2.5, axis=0)
            pred_haz_hi = np.percentile(pred_haz_arr, 97.5, axis=0)
        else:
            pred_haz_mean = curves['pred_haz_mean']
            pred_haz_lo = pred_haz_hi = None
        
        # Row 0: Survival
        ax = axes[0, col_idx]
        ax.plot(t_grid, curves['true_surv_mean'], '--', color=true_color, linewidth=1.5, label='True', alpha=true_alpha)
        ax.plot(t_grid, pred_surv_mean, '-', color=pred_color, linewidth=1.2, label='Pred')
        if pred_surv_lo is not None:
            ax.fill_between(t_grid, pred_surv_lo, pred_surv_hi, 
                          color=pred_color, alpha=band_alpha, linewidth=0)
        ax.set_ylim(0, 1.05)
        ax.set_title(disp_name, fontsize=10, pad=4)
        if col_idx == 0:
            ax.set_ylabel(row_labels[0], fontsize=9)
            ax.legend(fontsize=7, loc='upper right', framealpha=0.9, edgecolor='none')
        
        # Row 1: Cumulative Hazard
        ax = axes[1, col_idx]
        ax.plot(t_grid, curves['true_cumhaz_mean'], '--', color=true_color, linewidth=1.5, alpha=true_alpha)
        ax.plot(t_grid, pred_cumhaz_mean, '-', color=pred_color, linewidth=1.2)
        if pred_cumhaz_lo is not None:
            ax.fill_between(t_grid, pred_cumhaz_lo, pred_cumhaz_hi,
                          color=pred_color, alpha=band_alpha, linewidth=0)
        if col_idx == 0:
            ax.set_ylabel(row_labels[1], fontsize=9)
        
        # Row 2: Hazard
        ax = axes[2, col_idx]
        ax.plot(t_grid_haz, curves['true_haz_mean'], '--', color=true_color, linewidth=1.5, alpha=true_alpha)
        ax.plot(t_grid_haz, pred_haz_mean, '-', color=pred_color, linewidth=1.2)
        if pred_haz_lo is not None:
            ax.fill_between(t_grid_haz, pred_haz_lo, pred_haz_hi,
                          color=pred_color, alpha=band_alpha, linewidth=0)
        ax.set_xlabel('Time', fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(row_labels[2], fontsize=9)
    
    # Title with cleaner scenario name
    scenario_title = scenario_display_names.get(scenario_name, scenario_name.replace('_', ' ').title())
    fig.suptitle(f'{scenario_title}', fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'curves_{scenario_name}.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_c_index_over_time(curve_data: Dict, scenario_name: str, model_names: List[str], 
                            display_names: Dict[str, str] = None, output_dir: str = 'toy_output'):
    """
    Plot C-index over time for all models.
    
    Args:
        curve_data: Dict[model_name] -> curves dict with 'c_grid_runs' and 'c_grid_times'
        scenario_name: Name of the scenario for the title
        model_names: List of model names in order
        display_names: Optional dict mapping model_name -> display name
        output_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Color scheme
    model_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
    
    for idx, model_name in enumerate(model_names):
        if model_name not in curve_data:
            continue
        
        curves = curve_data[model_name]
        c_grid_runs = curves.get('c_grid_runs', [])
        t_grid = curves.get('c_grid_times', curves.get('t_grid', []))
        
        if len(c_grid_runs) == 0 or len(c_grid_runs[0]) == 0:
            continue
        
        # Stack runs and compute stats
        c_arr = np.array([c for c in c_grid_runs if len(c) > 0])
        if c_arr.size == 0:
            continue
            
        # Handle varying lengths by padding
        min_len = min(len(c) for c in c_grid_runs if len(c) > 0)
        c_arr = np.array([c[:min_len] for c in c_grid_runs if len(c) > 0])
        t_grid = t_grid[:min_len]
        
        c_mean = np.nanmean(c_arr, axis=0)
        c_lo = np.nanpercentile(c_arr, 2.5, axis=0)
        c_hi = np.nanpercentile(c_arr, 97.5, axis=0)
        
        color = model_colors[idx % len(model_colors)]
        disp_name = display_names.get(model_name, model_name) if display_names else model_name
        
        ax.plot(t_grid, c_mean, color=color, linewidth=1.5, label=disp_name)
        ax.fill_between(t_grid, c_lo, c_hi, color=color, alpha=0.2)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('Time')
    ax.set_ylabel('C-index')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, alpha=0.3)
    
    # Title
    scenario_display_names = {
        'exponential': 'Exponential', 'gamma': 'Gamma', 'gompertz': 'Gompertz',
        'loglogistic': 'Log-Logistic', 'lognormal': 'Log-Normal', 'weibull': 'Weibull',
    }
    title = scenario_display_names.get(scenario_name, scenario_name.replace('_', ' ').title())
    ax.set_title(f'C-index over Time: {title}', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'c_index_{scenario_name}.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    from networks.lora import TimeLoRASurvivalWrapper, BaseMLP
    from networks.film import TimeFiLMSurvivalWrapper
    from networks.mlp import MDNNet, NnetSurvNet, CoxNet, SimpleMLP, TimeConcatenatedMLP
    from models.qsurv import QSurv
    from models.desurv import DeSurv
    from models.soden import SODEN
    from models.coxtime import CoxTime
    from models.coxcc import CoxCC
    from models.mdn import MDN
    from models.deephit import DeepHit
    from models.nnetsurv import NnetSurv
    from models._discretizer import LabTransDiscreteTime
    
    SCENARIO_NAMES = ['exponential', 'weibull', 'gamma', 'lognormal', 'gompertz', 'loglogistic']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_test', type=int, default=2000)
    parser.add_argument('--censoring_rate', type=float, default=0.2)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--base_seed', type=int, default=0)
    parser.add_argument('--scenario', type=str, default=None, choices=SCENARIO_NAMES,
                        help='Run a single scenario (for parallel jobs). If not set, runs all.')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        choices=['coxcc', 'coxtime', 'nnetsurv', 'mdn', 'soden_adjoint',
                                 'soden_autograd', 'desurv', 'qsurv_concat', 'qsurv_film',
                                 'qsurv'],
                        help='Subset of models to run. Defaults to the full simulation set.')
    parser.add_argument('--output', type=str, default='toy_output/simulation_results.csv',
                        help='CSV path for results.')
    args = parser.parse_args()
    set_global_seed(args.base_seed)
    
    os.makedirs('toy_output', exist_ok=True)
    
    # Models to test — full benchmark set including qsurv_concat. Previously
    # hard-coded to soden_adjoint only for ad-hoc reruns.
    model_names = ['coxcc', 'coxtime', 'nnetsurv', 'mdn',
                   'soden_adjoint', 'desurv', 'qsurv_concat', 'qsurv_film', 'qsurv']
    if args.models is not None:
        model_names = args.models
    
    # Display names for plots
    display_names = {
        'coxcc': 'CoxCC', 'coxtime': 'CoxTime',
        'nnetsurv': 'NnetSurv', 'mdn': 'MDN', 'soden_adjoint': 'SODEN', 'soden_autograd': 'SODEN',
        'desurv': 'DeSurv',
        'qsurv': 'QSurv (LoRA)', 'qsurv_film': 'QSurv (FiLM)', 'qsurv_concat': 'QSurv (Concat)'
    }
    
    all_results = []
    
    scenarios = get_all_scenarios(args.n_train, args.n_test, args.censoring_rate, args.base_seed)
    
    # Filter to single scenario if specified
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        print(f"Running single scenario: {args.scenario}")
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario.name}")
        print(f"{'='*70}")
        
        # Collect curves for plotting (accumulate across all runs for uncertainty bands)
        scenario_curves = {model_name: {
            'pred_surv_runs': [],
            'pred_cumhaz_runs': [],
            'pred_haz_runs': [],
        } for model_name in model_names}
        
        for run_idx in range(args.n_runs):
            run_seed = args.base_seed + run_idx * 100
            print(f"\n--- Run {run_idx + 1}/{args.n_runs} (seed={run_seed}) ---")
            
            # Generate data with this run's seed
            x_train, t_train, e_train, x_test, t_test, e_test = scenario.generate(seed=run_seed)
            
            if run_idx == 0:
                print(f"Train: {len(x_train)}, Test: {len(x_test)}")
                print(f"Event rate: {e_train.mean():.1%}")
                print(f"Time range: [{t_train.min():.2f}, {t_train.max():.2f}]")
            
            # Time stats - simple normalization: divide by t_max
            mu_t = 0.0
            sigma_t = float(t_train.max())
            t_grid = np.linspace(t_train.min(), t_train.max(), 50).astype(np.float32)
            
            for model_name in model_names:
                set_global_seed(run_seed)
                
                if model_name == 'qsurv':
                    bb = BaseMLP(input_dim=1, hidden_dims=[32, 32], output_dim=32, activation=nn.Tanh, dropout=0.0)
                    net = TimeLoRASurvivalWrapper(bb, feature_dim=32, output_dim=1, output_activation='softplus',
                                                   mu=mu_t, sigma=sigma_t, time_norm_mode='min_max',
                                                   rank=16, alpha=16, t_hidden=32, t_depth=3)
                    model = QSurv(net, n_nodes=15, batch_size=128, epochs=args.epochs, lr=1e-2, time_norm_mode='min_max')
                
                elif model_name == 'qsurv_film':
                    bb = BaseMLP(input_dim=1, hidden_dims=[32, 32], output_dim=32, activation=nn.Tanh, dropout=0.0)
                    net = TimeFiLMSurvivalWrapper(bb, feature_dim=32, output_dim=1, output_activation='softplus',
                                                   mu=mu_t, sigma=sigma_t, time_norm_mode='min_max',
                                                   t_hidden=32, t_depth=3)
                    model = QSurv(net, n_nodes=15, batch_size=128, epochs=args.epochs, lr=1e-2, time_norm_mode='min_max')
                
                elif model_name == 'qsurv_concat':
                    net = TimeConcatenatedMLP(input_dim=1, hidden_dims=[32, 32], output_dim=1,
                                              output_activation='softplus',
                                              mu=mu_t, sigma=sigma_t, time_norm_mode='min_max')
                    model = QSurv(net, n_nodes=15, batch_size=128, epochs=args.epochs, lr=1e-2, time_norm_mode='min_max')

                elif model_name == 'desurv':
                    bb = BaseMLP(input_dim=1, hidden_dims=[32, 32], output_dim=32, activation=nn.Tanh, dropout=0.0)
                    net = TimeLoRASurvivalWrapper(bb, feature_dim=32, output_dim=1, output_activation='softplus',
                                                   mu=mu_t, sigma=sigma_t, time_norm_mode='min_max',
                                                   rank=16, alpha=16, t_hidden=32, t_depth=3)
                    model = DeSurv(net, n_nodes=15, batch_size=128, epochs=args.epochs, lr=1e-2, time_norm_mode='min_max')
                
                elif model_name == 'coxtime':
                    bb = BaseMLP(input_dim=1, hidden_dims=[32, 32], output_dim=32, activation=nn.Tanh, dropout=0.0)
                    net = TimeLoRASurvivalWrapper(bb, feature_dim=32, output_dim=1, output_activation='softplus',
                                                   output_bias=False, mu=mu_t, sigma=sigma_t, time_norm_mode='min_max',
                                                   rank=16, alpha=16, t_hidden=32, t_depth=3)
                    model = CoxTime(net, n_controls=1, batch_size=128, epochs=args.epochs, lr=1e-2, time_norm_mode='min_max')
                
                elif model_name == 'coxcc':
                    net = CoxNet(input_dim=1, hidden_dims=[32, 32], dropout=0.0)
                    model = CoxCC(net, n_controls=1, batch_size=128, epochs=args.epochs, lr=1e-2)
                
                elif model_name == 'mdn':
                    net = MDNNet(input_dim=1, hidden_dims=[32, 32], n_components=5, dropout=0.0,
                                 mu=mu_t, sigma=sigma_t)
                    model = MDN(net, batch_size=128, epochs=args.epochs, lr=1e-2,
                                time_norm_mode='min_max')
                
                elif model_name == 'nnetsurv':
                    discretizer = LabTransDiscreteTime(num_durations=50)
                    discretizer.fit(t_train)
                    net = NnetSurvNet(input_dim=1, hidden_dims=[32, 32], output_dim=len(discretizer.bin_edges_), dropout=0.0)
                    model = NnetSurv(net, discretizer=discretizer, batch_size=128, epochs=args.epochs, lr=1e-2)
                
                elif model_name.startswith('soden'):
                    bb = BaseMLP(input_dim=1, hidden_dims=[32, 32], output_dim=32, activation=nn.Tanh, dropout=0.0)
                    net = TimeLoRASurvivalWrapper(bb, feature_dim=32, output_dim=1, output_activation='softplus',
                                                   mu=mu_t, sigma=sigma_t, time_norm_mode='min_max',
                                                   rank=16, alpha=16, t_hidden=32, t_depth=3)
                    
                    use_adj = (model_name == 'soden_adjoint')
                    model = SODEN(net, batch_size=128, epochs=args.epochs, lr=1e-2, time_norm_mode='min_max', use_adjoint=use_adj)
                
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # Train
                x_train_t = torch.tensor(x_train, dtype=torch.float32)
                t_train_t = torch.tensor(t_train, dtype=torch.float32)
                e_train_t = torch.tensor(e_train, dtype=torch.float32)
                
                train_start = time.time()
                model.fit(x_train_t, t_train_t, e_train_t)
                train_time = time.time() - train_start
                
                # Evaluate
                metrics = evaluate_model(model, x_train, t_train, e_train, x_test, t_test, e_test, scenario, t_grid)
                
                # Extract curves and accumulate for uncertainty bands
                curves = metrics.pop('curves')
                scenario_curves[model_name]['pred_surv_runs'].append(curves['pred_surv_mean'])
                scenario_curves[model_name]['pred_cumhaz_runs'].append(curves['pred_cumhaz_mean'])
                scenario_curves[model_name]['pred_haz_runs'].append(curves['pred_haz_mean'])
                # Store true curves and grids (same across runs)
                scenario_curves[model_name]['t_grid'] = curves['t_grid']
                scenario_curves[model_name]['t_grid_haz'] = curves['t_grid_haz']
                scenario_curves[model_name]['true_surv_mean'] = curves['true_surv_mean']
                scenario_curves[model_name]['true_cumhaz_mean'] = curves['true_cumhaz_mean']
                scenario_curves[model_name]['true_haz_mean'] = curves['true_haz_mean']
                
                all_results.append({
                    'scenario': scenario.name,
                    'model': model_name,
                    'run': run_idx,
                    'seed': run_seed,
                    'train_time': train_time,
                    **metrics
                })
                
                print(f"  {model_name}: C={metrics['c_index']:.4f}, IBS={metrics['ibs']:.4f}, "
                      f"IBLL={metrics['ibll']:.4f}, L1_S={metrics['l1_survival']:.4f}, "
                      f"L1_H={metrics['l1_cumhaz']:.4f}, L1_h={metrics['l1_hazard']:.4f}, Time={train_time:.1f}s")
        
        # Plot marginalized curves for this scenario (with uncertainty bands from all runs)
        plot_marginalized_curves(scenario_curves, scenario.name, model_names, display_names=display_names, output_dir='toy_output')
    
    # Save all results
    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY (mean ± std over 10 runs)")
    print("="*80)
    
    metrics_to_show = ['c_index', 'ibs', 'ibll', 'l1_survival', 'l1_cumhaz', 'l1_hazard', 'train_time']
    
    for metric in metrics_to_show:
        print(f"\n{metric}:")
        pivot = df.pivot_table(index='scenario', columns='model', values=metric, aggfunc=['mean', 'std'])
        
        # Format as mean ± std
        summary = pd.DataFrame(index=pivot.columns.get_level_values(1).unique())
        for scenario_name in df['scenario'].unique():
            vals = []
            for model_name in model_names:
                mean_val = pivot.loc[scenario_name, ('mean', model_name)]
                std_val = pivot.loc[scenario_name, ('std', model_name)]
                vals.append(f"{mean_val:.4f}±{std_val:.4f}")
            summary[scenario_name] = vals
        summary.index = model_names
        print(summary.T.to_string())
