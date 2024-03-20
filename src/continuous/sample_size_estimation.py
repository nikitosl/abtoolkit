import numpy as np
from scipy import stats


def calculate_sample_size_by_mde(std, alpha, power, mde):
    """
    Calculate sample size need for significant ttest using std, alpha, power, mde
    :param std: tested variable std
    :param alpha: level of significance of A/B test
    :param power: probability of observing a statistically significant result at level alpha
    if a true effect of a certain magnitude is present
    :param mde: minimal detectable effect, the difference in results to detect
    :return: sample size needed for each group
    """

    z = stats.norm.ppf(q=1 - alpha / 2) + stats.norm.ppf(q=power)
    size = 2 * (std * z / mde) ** 2

    return round(size) + 1


def calculate_mde_by_sample_size(std, alpha, power, sample_size):
    """
    Calculate sample size need for significant ttest using std, alpha, power, mde
    :param std: tested variable std
    :param alpha: level of significance of A/B test
    :param power: probability of observing a statistically significant result at level alpha
    if a true effect of a certain magnitude is present
    :param sample_size: sample size needed for each group
    :return: minimal detectable effect, the difference in results to detect
    """
    z = stats.norm.ppf(q=1 - alpha / 2) + stats.norm.ppf(q=power)
    mde = std * z / np.sqrt(sample_size / 2)

    return mde
