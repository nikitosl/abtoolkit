"""
Simulates AA and AB tests to estimate test power and alpha
"""

from typing import List, Literal

import numpy as np

from abtoolkit.discrete.stattests import conversion_ztest
from abtoolkit.discrete.stattests import chi_square_test
from abtoolkit.discrete.stattests import bayesian_test
from abtoolkit.utils import BaseSimulationClass


class StatTestsSimulation(BaseSimulationClass):
    """
    Class simulates AA and AB tests to estimate test power, builds p-value distribution plot
    """

    def __init__(
        self,
        count: int,
        objects_num: int,
        alternative: Literal["less", "greater", "two-sided"],
        stattests_list: List[str],
        treatment_sample_size: int,
        treatment_split_proportion: float,
        experiments_num: int,
        mde: float,
        alpha_level: float = 0.05,
        power: float = 0.8,
        bayesian_prior_positives: int = 1,
        bayesian_prior_negatives: int = 1,
    ):
        """
        Simulates AA and AB tests for given stat-tests. Prints result (alpha and power) for each test
        and builds plot for p-value distributions.

        :param count: number of positive samples in dataset
        :param objects_num: number of all samples in dataset
        :param stattests_list: list of stat-tests for estimation
        :param treatment_sample_size: number of examples in test group to sample from variables in each iteration
        :param experiments_num: number of experiments to perform for each stat-test
        :param mde: minimal detectable effect, used to perform AB test (add to test variable)
        :param treatment_split_proportion: proportion of ab split for test group (50/50 -> 0.5, 80/20 -> 0.2, ...)
        :param alpha_level: test alpha-level
        :param power: test power
        :param bayesian_prior_positives: prior positive examples for bayesian stattest (default = 1)
        :param bayesian_prior_negatives: prior negative examples for bayesian stattest (default = 1)
        """
        super().__init__(
            treatment_sample_size=treatment_sample_size,
            treatment_split_proportion=treatment_split_proportion,
            alternative=alternative,
            mde=mde,
            power=power,
            alpha_level=alpha_level,
            stattests_list=stattests_list,
            experiments_num=experiments_num,
        )

        self.count = count
        self.objects_num = objects_num
        self.p = self.count / self.objects_num
        self.stattests_func_map = {
            "conversion_ztest": self.simulate_conversion_ztest,
            "bayesian_test": self.simulate_bayesian_test,
            "chi_square_test": self.simulate_chi_square_test,
        }
        self.bayesian_prior_positives = bayesian_prior_positives
        self.bayesian_prior_negatives = bayesian_prior_negatives

    def simulate_conversion_ztest(self, mde: float) -> float:
        """
        Simulate ttest
        :param mde: minimal detectable effect, to sum with test variable
        :return: p_value
        """

        control_count = np.random.binomial(n=self.control_sample_size, p=self.p)
        treatment_count = np.random.binomial(n=self.treatment_sample_size, p=self.p + mde)

        return conversion_ztest(
            control_count, self.control_sample_size, treatment_count, self.treatment_sample_size, self.alternative
        )

    def simulate_chi_square_test(self, mde: float) -> float:
        """
        Simulate chi-square test
        :param mde: minimal detectable effect, to sum with test variable
        :return: p_value
        """

        control_count = np.random.binomial(n=self.control_sample_size, p=self.p)
        treatment_count = np.random.binomial(n=self.treatment_sample_size, p=self.p + mde)

        return chi_square_test(control_count, self.control_sample_size, treatment_count, self.treatment_sample_size)

    def simulate_bayesian_test(self, mde: float) -> float:
        """
        Simulate bayesian test
        :param mde: minimal detectable effect, to sum with test variable
        :return: p_value
        """

        control_count = np.random.binomial(n=self.control_sample_size, p=self.p)
        treatment_count = np.random.binomial(n=self.treatment_sample_size, p=self.p + mde)

        return 1 - bayesian_test(
            control_count,
            self.control_sample_size,
            treatment_count,
            self.treatment_sample_size,
            self.alternative,
            self.bayesian_prior_positives,
            self.bayesian_prior_negatives,
        )
