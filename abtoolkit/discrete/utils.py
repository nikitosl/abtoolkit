"""
Utils functions for discrete variable analysis
"""

import numpy as np
from scipy import stats


def calculate_sample_size(p, alpha, power, mde) -> int:
    """
    Calculate sample size need for significant test using p, alpha, power, mde
    :param p: probability of positive sample
    :param alpha: level of significance of A/B test
    :param power: probability of observing a statistically significant result at level alpha
    if a true effect of a certain magnitude is present
    :param mde: minimal detectable effect, the difference in results to detect
    :return: sample size needed for each group
    """
    z = stats.norm.ppf(q=1 - alpha / 2) + stats.norm.ppf(q=power)
    size = 2 * p * (1 - p) * (z / mde) ** 2
    return round(size) + 1


def estimate_ci_binomial(p, sample_size, alpha):
    """
    Confidence interval for Binomial variable
    :param p: probability of positive
    :param sample_size: number of samples
    :param alpha: alpha level
    :return: low confidence interval value, high confidence interval value
    """
    t = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    std_n = np.sqrt(p * (1 - p) / sample_size)
    return p - t * std_n, p + t * std_n
