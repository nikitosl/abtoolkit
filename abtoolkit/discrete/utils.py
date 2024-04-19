"""
Utils functions for discrete variable analysis
"""

from typing import Literal

import numpy as np
from scipy import stats


def estimate_sample_size_by_mde(
    p: float,
    alpha: float,
    power: float,
    mde: float,
    alternative: Literal["less", "greater", "two-sided"],
) -> int:
    """
    Estimate sample size need for significant test using p, alpha, power, mde
    :param p: probability of positive sample
    :param alpha: level of significance of A/B test
    :param power: probability of observing a statistically significant result at level alpha
    if a true effect of a certain magnitude is present
    :param mde: minimal detectable effect, the difference in results to detect
    :param alternative: alternative hypothesis ("less", "greater" or "two-sided").
    * 'two-sided' : means are equal;
    * 'less': the mean of the control sample is less than the mean of the test sample;
    * 'greater': the mean of the control sample is greater than the mean of the test sample;
    :return: sample size needed for each group
    """

    # Todo: When alternative = "less" or "greater" we get undersampling (power less then 0.8).
    if alternative == "two-sided":
        pass

    alpha = alpha / 2

    z = stats.norm.ppf(q=1 - alpha) + stats.norm.ppf(q=power)
    size = 2 * p * (1 - p) * (z / mde) ** 2
    return round(size) + 1


def estimate_mde_by_sample_size(
    p: float,
    alpha: float,
    power: float,
    sample_size: int,
    alternative: Literal["less", "greater", "two-sided"],
) -> float:
    """
    Estimate MDE by need for significant test using p, alpha, power, sample_size
    :param p: probability of positive sample
    :param alpha: level of significance of A/B test
    :param power: probability of observing a statistically significant result at level alpha
    if a true effect of a certain magnitude is present
    :param sample_size: number of users in each group (test and control)
    :param alternative: alternative hypothesis ("less", "greater" or "two-sided").
    * 'two-sided' : means are equal;
    * 'less': the mean of the control sample is less than the mean of the test sample;
    * 'greater': the mean of the control sample is greater than the mean of the test sample;
    :return: minimum detectable effect
    """

    # Todo: When alternative = "less" or "greater" we get undersampling (power less then 0.8).
    if alternative == "two-sided":
        pass

    alpha = alpha / 2

    z = stats.norm.ppf(q=1 - alpha) + stats.norm.ppf(q=power)
    # size = 2 * p * (1 - p) * (z / mde) ** 2

    mde = z / np.sqrt(sample_size / (2 * p * (1 - p)))

    return mde


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
