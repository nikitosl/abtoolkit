"""
Stat tests for discrete variable analysis
"""

from typing import Literal

import numpy as np
from scipy import stats
from abtoolkit.discrete.utils import compare_beta_distributions


def conversion_ztest(
    control_count: int,
    control_objects_num: int,
    test_count: int,
    test_objects_num: int,
    alternative: Literal["less", "greater"],
) -> float:
    """
    Conversion z-test
    :param control_count: number of positive samples in control group
    :param control_objects_num: number of all samples in control group
    :param test_count: number of positive samples in test group
    :param test_objects_num: number of all samples in test group
    :param alternative: alternative hypothesis ("less", "greater" or "two-sided").
    * 'two-sided' : conversions are equal;
    * 'less': the conversion of the control sample is less than the mean of the test sample;
    * 'greater': the conversion of the control sample is greater than the mean of the test sample;
    :return: p-value
    """

    diff = control_count / control_objects_num - test_count / test_objects_num
    p_pooled = (control_count + test_count) / (control_objects_num + test_objects_num)
    std_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1 / control_objects_num + 1 / test_objects_num))
    z_stat = diff / std_diff

    if alternative == "less":
        p_value = stats.norm.cdf(z_stat)
    elif alternative == "greater":
        p_value = stats.norm.sf(z_stat)
    elif alternative == "two-sided":
        p_value = stats.norm.sf(np.abs(z_stat)) * 2
    else:
        raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")

    return p_value


def bayesian_test(
    control_count: int,
    control_objects_num: int,
    test_count: int,
    test_objects_num: int,
    alternative: Literal["less", "greater"],
    prior_positives_count: int = 1,
    prior_negatives_count: int = 1,
) -> float:
    """
    Bayesian Test
    :param control_count: posterior number of positive samples in control group
    :param control_objects_num: posterior number of all samples in control group
    :param test_count: posterior number of positive samples in test group
    :param test_objects_num: posterior number of all samples in test group
    :param alternative: alternative hypothesis ("less", "greater" or "two-sided").
    * 'less': the conversion of the control sample is less than the convertion in the test sample;
    * 'greater': the conversion of the control sample is greater than the conversion in the test sample;
    :param prior_positives_count: prior number of positive samples (alpha in Beta distribution)
    :param prior_negatives_count: prior number of negative samples (beta in Beta distribution)
    :return: probability that difference between distributions not exists (probability of null hypothesis)
    """

    # calculate posterior params for distributions
    alpha_1, beta_1 = test_count + prior_positives_count, test_objects_num - test_count + prior_negatives_count
    alpha_2, beta_2 = control_count + prior_positives_count, control_objects_num - control_count + prior_negatives_count

    if alternative == "less":
        return compare_beta_distributions(alpha_1, beta_1, alpha_2, beta_2)

    return compare_beta_distributions(alpha_2, beta_2, alpha_1, beta_1)
