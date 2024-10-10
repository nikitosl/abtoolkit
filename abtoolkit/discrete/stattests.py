"""
Stat tests for discrete variable analysis
"""

from math import lgamma
from typing import Literal

import numpy as np
from scipy import stats


def conversion_ztest(
    control_count: int,
    control_objects_num: int,
    test_count: int,
    test_objects_num: int,
    alternative: Literal["less", "greater", "two-sided"],
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
) -> float:
    """
    Bayesian Test
    :param control_count: number of positive samples in control group
    :param control_objects_num: number of all samples in control group
    :param test_count: number of positive samples in test group
    :param test_objects_num: number of all samples in test group
    :return: probability that difference between distributions not exists (probability of null hypothesis)
    """

    def diff_probability(beta1, beta2):
        a1, b1 = beta1.args[0], beta1.args[1]
        a2, b2 = beta2.args[0], beta2.args[1]

        ap = np.exp(lgamma(a1 + b1) + lgamma(a1 + a2) - (lgamma(a1 + b1 + a2) + lgamma(a1)))

        s = 0
        while b2 > 1:
            b2 -= 1
            num = lgamma(a1 + a2) + lgamma(b1 + b2) + lgamma(a1 + b1) + lgamma(a2 + b2)
            den = lgamma(a1) + lgamma(b1) + lgamma(a2) + lgamma(b2) + lgamma(a1 + b1 + a2 + b2)
            s += np.exp(num - den) / b2

        return ap + s

    beta_control = stats.beta(control_count + 1, control_objects_num - control_count + 1)
    beta_test = stats.beta(test_count + 1, test_objects_num - test_count + 1)

    return 1 - diff_probability(beta_test, beta_control)
