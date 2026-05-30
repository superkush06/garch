"""GARCH-family conditional variance recursions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GARCH11:
    """GARCH(1,1): sigma^2_t = omega + alpha * eps_{t-1}^2 + beta * sigma^2_{t-1}.

    Stationarity: alpha + beta < 1, omega > 0, alpha >= 0, beta >= 0.
    """
    omega: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.omega <= 0:
            raise ValueError("omega must be > 0")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha, beta must be >= 0")
        if self.alpha + self.beta >= 1.0:
            raise ValueError(f"non-stationary: alpha+beta = {self.alpha + self.beta}")

    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.alpha - self.beta)

    def filter(self, eps: np.ndarray) -> np.ndarray:
        """Compute conditional variance series from residuals."""
        T = eps.shape[0]
        sig2 = np.zeros(T)
        sig2[0] = self.unconditional_variance()
        for t in range(1, T):
            sig2[t] = (self.omega + self.alpha * eps[t - 1] ** 2
                       + self.beta * sig2[t - 1])
        return sig2

    def simulate(self, n: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Simulate T returns + variances starting from unconditional var."""
        rng = np.random.default_rng(seed)
        eps = np.zeros(n)
        sig2 = np.zeros(n)
        sig2[0] = self.unconditional_variance()
        eps[0] = math.sqrt(sig2[0]) * rng.standard_normal()
        for t in range(1, n):
            sig2[t] = (self.omega + self.alpha * eps[t - 1] ** 2
                       + self.beta * sig2[t - 1])
            eps[t] = math.sqrt(sig2[t]) * rng.standard_normal()
        return eps, sig2


@dataclass
class GJRGARCH:
    """GJR-GARCH(1,1,1): allows asymmetric (leverage) effect.

    sigma^2_t = omega + alpha*eps^2 + gamma*eps^2*I(eps<0) + beta*sigma^2_{t-1}.
    Positive gamma -> negative returns increase vol more than positive
    (the "leverage effect" in equities).
    """
    omega: float
    alpha: float
    gamma: float
    beta: float

    def __post_init__(self) -> None:
        if self.omega <= 0:
            raise ValueError("omega must be > 0")
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("alpha, beta, gamma must be >= 0")
        if self.alpha + 0.5 * self.gamma + self.beta >= 1.0:
            raise ValueError("non-stationary GJR")

    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.alpha - 0.5 * self.gamma - self.beta)

    def filter(self, eps: np.ndarray) -> np.ndarray:
        T = eps.shape[0]
        sig2 = np.zeros(T)
        sig2[0] = self.unconditional_variance()
        for t in range(1, T):
            asym = self.gamma * eps[t - 1] ** 2 * (eps[t - 1] < 0)
            sig2[t] = (self.omega + self.alpha * eps[t - 1] ** 2 + asym
                       + self.beta * sig2[t - 1])
        return sig2

    def simulate(self, n: int, *, seed: int = 0):
        rng = np.random.default_rng(seed)
        eps = np.zeros(n)
        sig2 = np.zeros(n)
        sig2[0] = self.unconditional_variance()
        eps[0] = math.sqrt(sig2[0]) * rng.standard_normal()
        for t in range(1, n):
            asym = self.gamma * eps[t - 1] ** 2 * (eps[t - 1] < 0)
            sig2[t] = (self.omega + self.alpha * eps[t - 1] ** 2 + asym
                       + self.beta * sig2[t - 1])
            eps[t] = math.sqrt(sig2[t]) * rng.standard_normal()
        return eps, sig2


@dataclass
class EGARCH:
    """EGARCH(1,1): log-variance recursion with leverage.

    log(sigma^2_t) = omega + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1}
                      + beta*log(sigma^2_{t-1}),
    where z_t = eps_t / sigma_t. Captures sign asymmetry without
    parameter positivity constraints.
    """
    omega: float
    alpha: float
    gamma: float
    beta: float

    def __post_init__(self) -> None:
        if abs(self.beta) >= 1.0:
            raise ValueError(f"non-stationary EGARCH: |beta| = {abs(self.beta)}")

    def filter(self, eps: np.ndarray) -> np.ndarray:
        T = eps.shape[0]
        log_sig2 = np.zeros(T)
        log_sig2[0] = self.omega / max(1e-9, 1.0 - self.beta)  # uncond log-var
        E_abs_z = math.sqrt(2.0 / math.pi)  # for Gaussian z
        for t in range(1, T):
            z_prev = eps[t - 1] / math.sqrt(math.exp(log_sig2[t - 1]))
            log_sig2[t] = (self.omega
                           + self.alpha * (abs(z_prev) - E_abs_z)
                           + self.gamma * z_prev
                           + self.beta * log_sig2[t - 1])
        return np.exp(log_sig2)

    def simulate(self, n: int, *, seed: int = 0):
        rng = np.random.default_rng(seed)
        log_sig2 = np.zeros(n)
        log_sig2[0] = self.omega / max(1e-9, 1.0 - self.beta)
        eps = np.zeros(n)
        eps[0] = math.sqrt(math.exp(log_sig2[0])) * rng.standard_normal()
        E_abs_z = math.sqrt(2.0 / math.pi)
        for t in range(1, n):
            z_prev = eps[t - 1] / math.sqrt(math.exp(log_sig2[t - 1]))
            log_sig2[t] = (self.omega
                           + self.alpha * (abs(z_prev) - E_abs_z)
                           + self.gamma * z_prev
                           + self.beta * log_sig2[t - 1])
            eps[t] = math.sqrt(math.exp(log_sig2[t])) * rng.standard_normal()
        return eps, np.exp(log_sig2)
