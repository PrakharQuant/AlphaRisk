import numpy as np
from scipy.optimize import brentq

def calculate_as_index(returns):
    """Calculates the Aumann-Serrano Risk Index."""
    if returns is None or len(returns) < 2: return np.nan
    returns = np.array(returns).flatten()
    mu = np.mean(returns)
    if mu <= 0: return np.inf

    def objective(alpha):
        return np.mean(np.exp(np.clip(-alpha * returns, -100, 100))) - 1

    try:
        return 1 / brentq(objective, 1e-10, 100)
    except (ValueError, RuntimeError):
        return np.nan

def calculate_fh_index(returns):
    """Calculates the Foster-Hart Risk Index."""
    if returns is None or len(returns) < 2: return np.nan
    returns = np.array(returns).flatten()
    mu = np.mean(returns)
    max_loss = -np.min(returns)
    if mu <= 0: return np.inf
    if max_loss <= 0: return 0.0

    def objective(R):
        return np.mean(np.log(1 + returns / R))

    try:
        return brentq(objective, max_loss + 1e-8, max_loss * 100_000)
    except (ValueError, RuntimeError):
        return np.nan
