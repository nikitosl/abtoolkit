"""
Utils for stat analysis
"""

from typing import Literal, List
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from abtoolkit.discrete.utils import estimate_ci_binomial


def generate_data(
    size: int,
    distribution_type: Literal["cont", "disc"],
    add_index: bool = True,
    index: Union[pd.Index, pd.MultiIndex] = None,
) -> pd.Series:
    """
    Generates data for continuous variables test. Regression technique require special multi-index [entity, dt].
    :param size: sample size for generation
    :param distribution_type: type of variable for generation
    * cont - continuous variable from normal distribution with mean=0 and std=3
    * disc - discrete variable from binomial distribution with p=0.2
    :param add_index: whether to add multiindex or not
    :param index: index for generated sample if add_index is true.
    If add_index is false then multiindex will be generated
    :return:  sample
    """

    if distribution_type == "cont":
        variable = np.random.normal(0, 3, size=size)
    elif distribution_type == "disc":
        variable = np.random.choice([0, 1], size=size, p=[0.8, 0.2])
    else:
        raise ValueError("type should take one of next values: ['cont', 'disc']")

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


class BaseSimulationClass:
    """
    Virtual class for AA and AB tests simulation
    """

    def __init__(
        self,
        alternative: Literal["less", "greater", "two-sided"],
        stattests_list: List[str],
        sample_size: int,
        experiments_num: int,
        mde: float,
        alpha_level: float = 0.05,
        power: float = 0.8,
    ):
        self.sample_size = sample_size
        self.alternative = alternative
        self.mde = mde
        self.power = power
        self.alpha_level = alpha_level
        self.stattests_list = stattests_list
        self.experiments_num = experiments_num
        self.info = {}
        self.stattests_func_map = {}

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
                # fail
                print("\033[91m" + f"'{test_name}'; alpha={a} [{aci1}; {aci2}], power={p} [{pci1}; {pci2}]" + "\033[0m")
            else:
                # success
                print("\033[92m" + f"'{test_name}'; alpha={a} [{aci1}; {aci2}], power={p} [{pci1}; {pci2}]" + "\033[0m")

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
