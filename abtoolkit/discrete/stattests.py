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
    treatment_count: int,
    treatment_objects_num: int,
    alternative: Literal["less", "greater"],
) -> float:
    """
    Conversion z-test
    :param control_count: number of positive samples in control group
    :param control_objects_num: number of all samples in control group
    :param treatment_count: number of positive samples in test group
    :param treatment_objects_num: number of all samples in test group
    :param alternative: alternative hypothesis ("less", "greater" or "two-sided").
    * 'two-sided' : conversions are equal;
    * 'less': the conversion of the control sample is less than the mean of the test sample;
    * 'greater': the conversion of the control sample is greater than the mean of the test sample;
    :return: p-value
    """

    diff = control_count / control_objects_num - treatment_count / treatment_objects_num
    p_pooled = (control_count + treatment_count) / (control_objects_num + treatment_objects_num)
    std_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1 / control_objects_num + 1 / treatment_objects_num))
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
    treatment_count: int,
    treatment_objects_num: int,
    alternative: Literal["less", "greater"],
    prior_positives_count: int = 1,
    prior_negatives_count: int = 1,
) -> float:
    """
    Bayesian Test
    :param control_count: posterior number of positive samples in control group
    :param control_objects_num: posterior number of all samples in control group
    :param treatment_count: posterior number of positive samples in test group
    :param treatment_objects_num: posterior number of all samples in test group
    :param alternative: alternative hypothesis ("less", "greater" or "two-sided").
    * 'less': the conversion of the control sample is less than the convertion in the test sample;
    * 'greater': the conversion of the control sample is greater than the conversion in the test sample;
    :param prior_positives_count: prior number of positive samples (alpha in Beta distribution)
    :param prior_negatives_count: prior number of negative samples (beta in Beta distribution)
    :return: probability that difference between distributions exists
    """

    # calculate posterior params for distributions
    alpha_1 = treatment_count + prior_positives_count
    beta_1 = treatment_objects_num - treatment_count + prior_negatives_count
    alpha_2 = control_count + prior_positives_count
    beta_2 = control_objects_num - control_count + prior_negatives_count

    probability = compare_beta_distributions(alpha_1, beta_1, alpha_2, beta_2)

    if alternative == "greater":
        return probability

    return 1 - probability


def chi_square_test(
    control_count: int,
    control_objects_num: int,
    treatment_count: int,
    treatment_objects_num: int,
) -> float:
    """
    Chi-square test
    :param control_count: number of positive samples in control group
    :param control_objects_num: number of all samples in control group
    :param treatment_count: number of positive samples in test group
    :param treatment_objects_num: number of all samples in test group
    :return: p-value
    """

    control_negative_count = control_objects_num - control_count
    treatment_negative_count = treatment_objects_num - treatment_count

    if control_negative_count < 5 or control_count < 5 or treatment_negative_count < 5 or treatment_count < 5:
        raise ValueError("Too few samples for chi-square test (>= 5 in each case)")

    contingency_table = [[control_count, control_negative_count], [treatment_count, treatment_negative_count]]

    _, pvalue, _, _ = stats.chi2_contingency(contingency_table, correction=True)

    return pvalue
