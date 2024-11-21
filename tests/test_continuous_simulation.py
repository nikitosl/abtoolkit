import unittest
from abtoolkit.continuous.simulation import StatTestsSimulation
from abtoolkit.utils import generate_data


class TestStatTestsSimulation(unittest.TestCase):
    tests = ["ttest", "diff_ttest", "regression_test", "cuped_ttest",
             "did_regression_test", "additional_vars_regression_test"]

    def test_success(self):
        variable = generate_data(100, distribution_type="cont")
        previous_value = generate_data(100, distribution_type="cont", index=variable.index).rename("prev")

        experiments_num = 10

        sim = StatTestsSimulation(
            variable,
            stattests_list=self.tests,
            experiments_num=experiments_num,
            alternative="two-sided",
            treatment_sample_size=50,
            treatment_split_proportion=0.5,
            mde=10,
            alpha_level=0.05,
            power=0.8,

            previous_values=previous_value,
            cuped_covariant=previous_value,
            additional_vars=[previous_value],
        )
        info = sim.run()

        self.assertTrue(len(self.tests) == len(info.keys()),
                        "Number of tests doesn't match with number of result dicts")
        for test in self.tests:
            self.assertTrue(test in info.keys(), f"Test '{test}' is not in result dict")
            test_info = info[test]
            self.assertTrue("alpha" in test_info, f"Alpha value not in result dict")
            self.assertTrue(0 <= test_info["alpha"] <= 1, f"Alpha value has wrong value: {test_info['alpha']}")

            self.assertTrue("alpha_ci" in test_info, f"Alpha CI value not in result dict")
            self.assertTrue(len(test_info["alpha_ci"]) == 2,
                            f"Wrong confidence interval for alpha: {test_info['alpha_ci']}")

            self.assertTrue("power" in test_info, f"Power value not in result dict")
            self.assertTrue(0 <= test_info["power"] <= 1, f"Power value has wrong value: {test_info['power']}")

            self.assertTrue("power_ci" in test_info, f"Power CI value not in result dict")
            self.assertTrue(len(test_info["power_ci"]) == 2,
                            f"Wrong confidence interval for power: {test_info['power_ci']}")

            self.assertTrue("aa_pvalues" in test_info, f"AA p-values not in result dict")
            self.assertTrue(len(test_info["aa_pvalues"]) == experiments_num,
                            f"Number of p-values in AA test doesn't match with number of experiments")

            self.assertTrue("ab_pvalues" in test_info, f"AB p-values not in result dict")
            self.assertTrue(len(test_info["ab_pvalues"]) == experiments_num,
                            f"Number of p-values in AB test doesn't match with number of experiments")

    def test_unbalanced_ab(self):
        variable = generate_data(100, distribution_type="cont")
        previous_value = generate_data(100, distribution_type="cont", index=variable.index).rename("prev")

        experiments_num = 10

        sim = StatTestsSimulation(
            variable,
            stattests_list=self.tests,
            experiments_num=experiments_num,
            alternative="two-sided",
            treatment_sample_size=50,
            treatment_split_proportion=0.7,
            mde=10,
            alpha_level=0.05,
            power=0.8,

            previous_values=previous_value,
            cuped_covariant=previous_value,
            additional_vars=[previous_value],
        )
        info = sim.run()

        self.assertTrue(len(self.tests) == len(info.keys()),
                        "Number of tests doesn't match with number of result dicts")
        for test in self.tests:
            self.assertTrue(test in info.keys(), f"Test '{test}' is not in result dict")
            test_info = info[test]
            self.assertTrue("alpha" in test_info, f"Alpha value not in result dict")
            self.assertTrue(0 <= test_info["alpha"] <= 1, f"Alpha value has wrong value: {test_info['alpha']}")

            self.assertTrue("alpha_ci" in test_info, f"Alpha CI value not in result dict")
            self.assertTrue(len(test_info["alpha_ci"]) == 2,
                            f"Wrong confidence interval for alpha: {test_info['alpha_ci']}")

            self.assertTrue("power" in test_info, f"Power value not in result dict")
            self.assertTrue(0 <= test_info["power"] <= 1, f"Power value has wrong value: {test_info['power']}")

            self.assertTrue("power_ci" in test_info, f"Power CI value not in result dict")
            self.assertTrue(len(test_info["power_ci"]) == 2,
                            f"Wrong confidence interval for power: {test_info['power_ci']}")

            self.assertTrue("aa_pvalues" in test_info, f"AA p-values not in result dict")
            self.assertTrue(len(test_info["aa_pvalues"]) == experiments_num,
                            f"Number of p-values in AA test doesn't match with number of experiments")

            self.assertTrue("ab_pvalues" in test_info, f"AB p-values not in result dict")
            self.assertTrue(len(test_info["ab_pvalues"]) == experiments_num,
                            f"Number of p-values in AB test doesn't match with number of experiments")
