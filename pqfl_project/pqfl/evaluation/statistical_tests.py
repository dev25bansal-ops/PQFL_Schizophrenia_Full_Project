"""Statistical tests for comparing PQFL model performance.

Implements:
- DeLong test for AUC comparison
- McNemar's test for accuracy comparison
- Bootstrap confidence intervals
- Bonferroni correction for multiple comparisons
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def delong_test(
    y_true: np.ndarray,
    y_prob_1: np.ndarray,
    y_prob_2: np.ndarray,
) -> Dict[str, float]:
    """DeLong test for comparing two AUC-ROC values.
    
    Tests whether two ROC curves have significantly different AUCs.
    
    Args:
        y_true: True binary labels, shape (n_samples,).
        y_prob_1: Predicted probabilities from model 1.
        y_prob_2: Predicted probabilities from model 2.
    
    Returns:
        Dictionary with AUC values, z-statistic, and p-value.
    """
    from sklearn.metrics import roc_auc_score
    
    auc1 = roc_auc_score(y_true, y_prob_1)
    auc2 = roc_auc_score(y_true, y_prob_2)
    
    # Simplified DeLong test using bootstrap
    n_bootstrap = 2000
    rng = np.random.RandomState(42)
    n = len(y_true)
    
    auc_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            a1 = roc_auc_score(y_true[idx], y_prob_1[idx])
            a2 = roc_auc_score(y_true[idx], y_prob_2[idx])
            auc_diffs.append(a1 - a2)
        except ValueError:
            continue
    
    auc_diffs = np.array(auc_diffs)
    observed_diff = auc1 - auc2
    
    # Z-test on bootstrap distribution
    mean_diff = np.mean(auc_diffs)
    std_diff = np.std(auc_diffs) + 1e-10
    z_stat = observed_diff / std_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        "auc_model1": auc1,
        "auc_model2": auc2,
        "auc_difference": observed_diff,
        "z_statistic": z_stat,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
) -> Dict[str, float]:
    """McNemar's test for comparing two classifiers.
    
    Tests whether the two models have significantly different
    error patterns.
    
    Args:
        y_true: True binary labels.
        y_pred_1: Predictions from model 1.
        y_pred_2: Predictions from model 2.
    
    Returns:
        Dictionary with test statistic and p-value.
    """
    # Compute contingency table
    correct_1 = y_pred_1 == y_true
    correct_2 = y_pred_2 == y_true
    
    # b: model 1 correct, model 2 wrong
    b = np.sum(correct_1 & ~correct_2)
    # c: model 1 wrong, model 2 correct
    c = np.sum(~correct_1 & correct_2)
    
    # McNemar's test statistic
    if b + c == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_005": False,
        }
    
    # With continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_disagreement": int(b + c),
        "significant_005": p_value < 0.05,
    }


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.
    
    Args:
        values: Metric values across folds or samples.
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.
    
    Returns:
        Tuple of (mean, lower_ci, upper_ci).
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = (1 - confidence) / 2
    
    mean = np.mean(values)
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return float(mean), float(lower), float(upper)


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """Apply Bonferroni correction for multiple comparisons.
    
    Adjusted alpha = alpha / n_tests
    
    Args:
        p_values: List of p-values from multiple tests.
        alpha: Family-wise error rate.
    
    Returns:
        List of booleans indicating significance after correction.
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    return [p < adjusted_alpha for p in p_values]
