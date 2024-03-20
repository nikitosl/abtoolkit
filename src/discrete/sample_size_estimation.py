from scipy import stats
import numpy as np


def estimate_ci_binomial(p, sample_size, alpha=0.05):
    """
    Confidence interval for Binomial variable
    :param p: probability of positive
    :param sample_size: number of samples
    :param alpha: alpha level
    :return: lower confidence interval value, higher confidence interval value
    """
    t = stats.norm.ppf(1 - alpha / 2, loc=0, scale=1)
    std_n = np.sqrt(p * (1 - p) / sample_size)
    return p - t * std_n, p + t * std_n
