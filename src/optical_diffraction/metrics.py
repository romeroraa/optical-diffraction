"""
Metrics for evaluating Monte Carlo simulation quality.

Includes Linfoot's criteria and error metrics.
"""

import numpy as np


def normalize(sigma, I):
    """
    Normalize two arrays to have unit L2 norm.

    Parameters
    ----------
    sigma : np.ndarray
        Reference signal
    I : np.ndarray
        Test signal

    Returns
    -------
    tuple of np.ndarray
        (normalized_sigma, normalized_I)
    """
    nsigma = sigma / np.sqrt(np.sum(sigma**2))
    nI = I / np.sqrt(np.sum(I**2))
    return nsigma, nI


def rst(sigma, I):
    """
    Relative structural content (Linfoot's criterion).

    Uses normalized data. Should equal 1.0 when both signals
    have unit L2 norm.

    Parameters
    ----------
    sigma : np.ndarray
        Reference signal
    I : np.ndarray
        Test signal

    Returns
    -------
    float
        Relative structural content
    """
    nsigma, nI = normalize(sigma, I)
    return np.sum(nI**2) / np.sum(nsigma**2)


def fidelity(sigma, I):
    """
    Fidelity metric (Linfoot's criterion).

    1.0 = perfect match.

    Parameters
    ----------
    sigma : np.ndarray
        Reference signal
    I : np.ndarray
        Test signal

    Returns
    -------
    float
        Fidelity (1.0 = perfect)
    """
    nsigma, nI = normalize(sigma, I)
    return 1.0 - np.sum((nI - nsigma)**2) / np.sum(nsigma**2)


def correlationquality(sigma, I):
    """
    Correlation quality metric (Linfoot's criterion).

    Parameters
    ----------
    sigma : np.ndarray
        Reference signal
    I : np.ndarray
        Test signal

    Returns
    -------
    float
        Correlation quality
    """
    nsigma, nI = normalize(sigma, I)
    return np.sum(nI * nsigma) / np.sum(nsigma**2)


def nrmse(sr, se):
    """
    Normalized root mean square error.

    Parameters
    ----------
    sr : array-like
        Observed data
    se : array-like
        Theoretical/reference data

    Returns
    -------
    float
        Normalized RMSE
    """
    denom = np.max(sr) - np.min(sr)
    if denom == 0:
        return 0.0
    return np.sqrt(np.mean((np.asarray(sr) - np.asarray(se))**2)) / denom


def rmse(observed, reference):
    """
    Root mean square error.

    Parameters
    ----------
    observed : np.ndarray
        Observed data
    reference : np.ndarray
        Reference/theoretical data

    Returns
    -------
    float
        RMSE
    """
    return np.sqrt(np.mean((observed - reference)**2))


def verify_linfoot_relationship(F, Q, S):
    """
    Verify that Linfoot's relationship F = 2Q - S holds.

    Parameters
    ----------
    F : float
        Fidelity
    Q : float
        Correlation quality
    S : float
        Relative structural content

    Returns
    -------
    float
        Absolute difference |F - (2Q - S)|
    """
    return np.abs(F - (2*Q - S))
