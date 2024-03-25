"""
Simulates AA and AB tests to estimate test power and alpha
"""

from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from abtoolkit.continuous.stattests import additional_vars_regression_test
from abtoolkit.continuous.stattests import cuped_ttest
from abtoolkit.continuous.stattests import did_regression_test
from abtoolkit.continuous.stattests import regression_test
from abtoolkit.continuous.stattests import ttest
from abtoolkit.discrete.utils import estimate_ci_binomial


class StatTestsSimulation:
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

        self.variable = variable

        self.alternative = alternative
        self.stattests_list = stattests_list
        self.experiments_num = experiments_num
        self.sample_size = sample_size
        self.mde = mde
        self.alpha_level = alpha_level
        self.power = power

        self.stattests_func_map = {
            "ttest": self.simulate_ttest,
            "diff_ttest": self.simulate_difference_ttest,
            "cuped_ttest": self.simulate_cuped,
            "regression_test": self.simulate_reg,
            "did_regression_test": self.simulate_reg_did,
            "additional_vars_regression_test": self.simulate_reg_add,
        }
        self.info = {}

        # Optional
        self.previous_values = previous_values
        self.cuped_covariant = cuped_covariant
        self.additional_vars = additional_vars

    def plot_p_values(self):
        """
        Plot p-values distribution for each test
        :return: None
        """
        if len(self.info) == 0:
            return

        x_axis = np.linspace(0, 1, 1000)
        for test, test_info in self.info.items():
            ab_pvalues = np.array(test_info["ab_pvalues"])
            y_axis = [np.mean(ab_pvalues < x) for x in x_axis]
            plt.plot(x_axis, y_axis, label=test)

        plt.plot([self.alpha_level, self.alpha_level], [0, 1], "--k", alpha=0.8)
        plt.plot([0, 1], [self.power, self.power], "--k", alpha=0.8)
        plt.title("P-Value Distribution for AB Simulation", size=12)
        plt.xlabel("p-value", size=10)
        plt.legend(fontsize=10)
        plt.grid()
        plt.show()

    def print_results(self):
        """
        Print simulation results for each test (alpha and power + confidence intervals)
        :return: None
        """
        for test_name, test_info in self.info.items():
            a, p = test_info["alpha"], test_info["power"]
            aci1, aci2 = round(test_info["alpha_ci"][0], 4), round(test_info["alpha_ci"][1], 4)
            pci1, pci2 = round(test_info["power_ci"][0], 4), round(test_info["power_ci"][1], 4)

            if (aci1 > self.alpha_level) or (self.power > pci2):
                print(
                    "\033[91m" + f"'{test_name}'; alpha={a} ci[{aci1}; {aci2}], power={p} [{pci1}; {pci2}]" + "\033[0m"
                )
            else:
                print(
                    "\033[92m" + f"'{test_name}'; alpha={a} ci[{aci1}; {aci2}], power={p} [{pci1}; {pci2}]" + "\033[0m"
                )

    def run(self):
        """
        Simulate all tests from 'self.stattests_list' by given data and save information to 'info' dictionary
        :return:
        """
        self.info = {}
        for stattest in self.stattests_list:
            self.simulate_test_by_name(stattest)
        return self.info

    def simulate_test_by_name(self, test_name: str):
        """
        Simulate AA and AB test and save results to 'info' dictionary
        :param test_name: name of test for simulation (ttest | cuped_ttest | regression_test | did_regression_test
                                                        | additional_vars_regression_test)
        :return: None
        """

        assert test_name in self.stattests_func_map, f"Given test_name {test_name} not found"
        stattest_func = self.stattests_func_map[test_name]

        test_success_no_effect_cnt = 0
        test_pvalues_no_effect = []
        test_success_effect_cnt = 0
        test_pvalues_effect = []

        for _ in tqdm(range(self.experiments_num), desc=f"Simulation test '{test_name}'"):
            p_value = stattest_func(mde=0)
            test_pvalues_no_effect.append(p_value)
            if p_value < self.alpha_level:
                test_success_no_effect_cnt += 1

            p_value = stattest_func(mde=self.mde)
            test_pvalues_effect.append(p_value)
            if p_value < self.alpha_level:
                test_success_effect_cnt += 1

        alpha = test_success_no_effect_cnt / self.experiments_num
        power = test_success_effect_cnt / self.experiments_num

        alpha_ci = estimate_ci_binomial(alpha, self.experiments_num, alpha=0.05)
        power_ci = estimate_ci_binomial(power, self.experiments_num, alpha=0.05)

        if test_name in self.info:
            del self.info[test_name]
        self.info[test_name] = {
            "alpha": alpha,
            "alpha_ci": alpha_ci,
            "power": power,
            "power_ci": power_ci,
            "aa_pvalues": test_pvalues_no_effect,
            "ab_pvalues": test_pvalues_effect,
        }

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
        test_pre_sample = self.variable.loc[test_index_sample]
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
