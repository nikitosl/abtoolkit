"""
Simulates AA and AB tests to estimate test power and alpha
"""

from typing import List, Literal

import numpy as np
import pandas as pd

from abtoolkit.continuous.stattests import additional_vars_regression_test
from abtoolkit.continuous.stattests import cuped_ttest
from abtoolkit.continuous.stattests import did_regression_test
from abtoolkit.continuous.stattests import regression_test
from abtoolkit.continuous.stattests import ttest
from abtoolkit.utils import BaseSimulationClass


class StatTestsSimulation(BaseSimulationClass):
    """
    Class simulates AA and AB tests to estimate test power, builds p-value distribution plot
    """

    def __init__(
        self,
        variable: pd.Series,
        alternative: Literal["less", "greater", "two-sided"],
        stattests_list: List[str],
        sample_size: int,
        experiments_num: int,
        mde: float,
        alpha_level: float = 0.05,
        power: float = 0.8,
        previous_values: pd.Series = None,
        cuped_covariant: pd.Series = None,
        additional_vars: List[pd.Series] = None,
    ):
        """
        Simulates AA and AB tests for given stat-tests. Prints result (alpha and power) for each test
        and builds plot for p-value distributions.

        :param variable: variable for simulation
        :param stattests_list: list of stat-tests for estimation
        :param sample_size: number of examples to sample from variables in each iteration
        :param experiments_num: number of experiments to perform for each stat-test
        :param mde: minimal detectable effect, used to perform AB test (add to test variable)
        :param alpha_level: test alpha-level
        :param power: test power
        :param previous_values: previous values of variable used to reduce variance and speedup
        test in difference-in-difference test
        :param cuped_covariant: covariant for variable used to reduce variance and speedup test
        in cuped test
        :param additional_vars: list of additional variables used to
        reduce variance of main variable and speedup test in 'regression_with_additional_variables' test
        """
        super().__init__(
            sample_size=sample_size,
            alternative=alternative,
            mde=mde,
            power=power,
            alpha_level=alpha_level,
            stattests_list=stattests_list,
            experiments_num=experiments_num,
        )

        self.variable = variable
        self.stattests_func_map = {
            "ttest": self.simulate_ttest,
            "diff_ttest": self.simulate_difference_ttest,
            "cuped_ttest": self.simulate_cuped,
            "regression_test": self.simulate_reg,
            "did_regression_test": self.simulate_reg_did,
            "additional_vars_regression_test": self.simulate_reg_add,
        }

        # Optional
        self.previous_values = previous_values
        self.cuped_covariant = cuped_covariant
        self.additional_vars = additional_vars

    def simulate_ttest(self, mde: float) -> float:
        """
        Simulate ttest
        :param mde: minimal detectable effect, to sum with test variable
        :return: p_value
        """
        control_sample = self.variable.sample(self.sample_size, replace=True)
        test_sample = self.variable.sample(self.sample_size, replace=True)
        test_sample += mde

        return ttest(control_sample, test_sample, self.alternative)

    def simulate_difference_ttest(self, mde: float) -> float:
        """
        Simulate ttest for difference between actual variable value and previous period variable value
        :param mde: minimal detectable effect, to sum with test variable
        :return: p_value
        """
        control_index_sample = self.variable.index[np.random.randint(0, len(self.variable), size=self.sample_size)]
        test_index_sample = self.variable.index[np.random.randint(0, len(self.variable), size=self.sample_size)]

        control_sample = self.variable.loc[control_index_sample]
        control_pre_sample = self.previous_values.loc[control_index_sample]
        test_sample = self.variable.loc[test_index_sample]
        test_pre_sample = self.previous_values.loc[test_index_sample]
        test_sample += mde

        return cuped_ttest(control_sample, control_pre_sample, test_sample, test_pre_sample, self.alternative)

    def simulate_cuped(self, mde: float) -> float:
        """
        Simulate CUPED ttest
        :param mde: minimal detectable effect, to sum with test variable
        :return: p_value
        """
        control_index_sample = self.variable.index[np.random.randint(0, len(self.variable), size=self.sample_size)]
        test_index_sample = self.variable.index[np.random.randint(0, len(self.variable), size=self.sample_size)]

        control_sample = self.variable.loc[control_index_sample]
        control_covariant_sample = self.cuped_covariant.loc[control_index_sample]
        test_sample = self.variable.loc[test_index_sample]
        test_covariant_sample = self.cuped_covariant.loc[test_index_sample]
        test_sample += mde

        return cuped_ttest(
            control_sample, control_covariant_sample, test_sample, test_covariant_sample, self.alternative
        )

    def simulate_reg(self, mde: float) -> float:
        """
        Simulate test using regression
        :param mde: minimal detectable effect, to sum with test variable
        :return: p_value
        """
        control_sample = self.variable.sample(self.sample_size, replace=True)
        test_sample = self.variable.sample(self.sample_size, replace=True)
        test_sample += mde

        return regression_test(control_sample, test_sample, self.alternative)

    def simulate_reg_did(self, mde: float) -> float:
        """
        Simulate test using regression with difference-in-difference technique
        :param mde: minimal detectable effect, to sum with test variable
        :return: p_value
        """
        control_index_sample = self.variable.index[np.random.randint(0, len(self.variable), size=self.sample_size)]
        test_index_sample = self.variable.index[np.random.randint(0, len(self.variable), size=self.sample_size)]

        control_sample = self.variable.loc[control_index_sample]
        control_previous_sample = self.previous_values.loc[control_index_sample]
        test_sample = self.variable.loc[test_index_sample]
        test_previous_sample = self.previous_values.loc[test_index_sample]
        test_sample += mde

        return did_regression_test(
            control_sample, control_previous_sample, test_sample, test_previous_sample, self.alternative
        )

    def simulate_reg_add(self, mde: float) -> float:
        """
        Simulate test using regression with additional variables
        :param mde: minimal detectable effect, to sum with test variable
        :return: p_value
        """
        control_index_sample = self.variable.index[np.random.randint(0, len(self.variable), size=self.sample_size)]
        test_index_sample = self.variable.index[np.random.randint(0, len(self.variable), size=self.sample_size)]

        control_sample = self.variable.loc[control_index_sample]
        control_add_samples = [a.loc[control_index_sample] for a in self.additional_vars]
        test_sample = self.variable.loc[test_index_sample]
        test_add_samples = [a.loc[test_index_sample] for a in self.additional_vars]
        test_sample += mde

        return additional_vars_regression_test(
            control_sample, control_add_samples, test_sample, test_add_samples, self.alternative
        )
