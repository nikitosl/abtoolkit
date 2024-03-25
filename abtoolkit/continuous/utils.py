"""
Utils functions for continuous variables
"""

from typing import Tuple, Literal

import numpy as np
import pandas as pd
from scipy import stats


def generate_data(size, add_index=True, index=None) -> pd.Series:
    """
    Generates data for continuous variables test. Regression technique require special multi-index [entity, dt].
    :param size: sample size for generation
    :param add_index: whether to add multiindex or not
    :param index: index for generated sample if add_index is true.
    If add_index is false then multiindex will be generated
    :return:  sample
    """

    variable = np.random.normal(0, 3, size=size)
    if add_index:
        if index is None:
            # Generate random index
            entity_index = np.random.choice([1, 2, 3], size=size)
            time_index = np.arange(0, size, 1)
            result_sr = pd.Series(variable, index=[entity_index, time_index])
        else:
            # Add existing index
            result_sr = pd.Series(variable, index=index)
    else:
        # Without index
        result_sr = pd.Series(variable)
    return result_sr


def estimate_confidence_interval(
        mean: float,
        std: float,
        sample_size: int,
        alpha: float,
        power: float,
        alternative: Literal["less", "greater", "two-sided"],
) -> Tuple[float, float]:
    """
    Confidence interval estimation
    :param mean: sample average value
    :param std: sample standard deviation
    :param sample_size: number of samples
    :param alpha: alpha-level
    :param power: power
    :param alternative: alternative hypothesis ("less", "greater" or "two-sided").
    * 'two-sided' : means are equal;
    * 'less': the mean of the control sample is less than the mean of the test sample;
    * 'greater': the mean of the control sample is greater than the mean of the test sample;
    :return: low confidence interval value, high confidence interval value
    """
    if alternative == "two-sided":
        alpha = alpha / 2

    z = stats.norm.ppf(q=1 - alpha) + stats.norm.ppf(q=power)
    se = z * std / np.sqrt(sample_size)

    return mean - se, mean + se


def estimate_sample_size_by_mde(
        std: float,
        alpha: float,
        power: float,
        mde: float,
        alternative: Literal["less", "greater", "two-sided"],
) -> int:
    """
    Calculate sample size need for significant ttest using std, alpha, power, mde
    :param std: tested variable std
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

    if alternative == "two-sided":
        alpha = alpha / 2

    z = stats.norm.ppf(q=1 - alpha) + stats.norm.ppf(q=power)
    size = 2 * (std * z / mde) ** 2

    return round(size) + 1


def estimate_mde_by_sample_size(
        std: float,
        alpha: float,
        power: float,
        sample_size: int,
        alternative: Literal["less", "greater", "two-sided"],
):
    """
    Calculate sample size need for significant ttest using std, alpha, power, mde
    :param std: tested variable std
    :param alpha: level of significance of A/B test
    :param power: probability of observing a statistically significant result at level alpha
    if a true effect of a certain magnitude is present
    :param sample_size: sample size needed for each group
    :param alternative: alternative hypothesis ("less", "greater" or "two-sided").
    * 'two-sided' : means are equal;
    * 'less': the mean of the control sample is less than the mean of the test sample;
    * 'greater': the mean of the control sample is greater than the mean of the test sample;
    :return: minimal detectable effect, the difference in results to detect
    """
    if alternative == "two-sided":
        alpha = alpha / 2

    z = stats.norm.ppf(q=1 - alpha) + stats.norm.ppf(q=power)
    mde = std * z / np.sqrt(sample_size / 2)

    return mde
