import unittest

from abtoolkit.continuous.stattests import regression_test
from abtoolkit.continuous.stattests import did_regression_test
from abtoolkit.continuous.stattests import additional_vars_regression_test
from abtoolkit.continuous.stattests import cuped_ttest
from abtoolkit.continuous.stattests import difference_ttest
from abtoolkit.continuous.stattests import ttest
from abtoolkit.utils import generate_data


class TestStatTests(unittest.TestCase):
    def test_ttest(self):
        test_sr = generate_data(100, distribution_type="cont")
        control_sr = generate_data(100, distribution_type="cont")
        p_value = ttest(control_sr, test_sr, "two-sided")
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_regression_test(self):
        test_sr = generate_data(100, distribution_type="cont")
        control_sr = generate_data(100, distribution_type="cont")
        p_value = regression_test(control_sr, test_sr, "two-sided")
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_diff_ttest(self):
        test_sr = generate_data(100, distribution_type="cont")
        test_pre_sr = generate_data(100, distribution_type="cont", index=test_sr.index).rename("var_1")
        control_sr = generate_data(100, distribution_type="cont")
        control_pre_sr = generate_data(100, distribution_type="cont", index=control_sr.index).rename("var_1")
        p_value = difference_ttest(control_pre_sr, control_sr, test_pre_sr, test_sr, "two-sided")
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_did_regression_test(self):
        test_pre_sr = generate_data(100, distribution_type="cont")
        test_post_sr = generate_data(100, distribution_type="cont")
        control_pre_sr = generate_data(100, distribution_type="cont")
        control_post_sr = generate_data(100, distribution_type="cont")
        p_value = did_regression_test(control_pre_sr, control_post_sr, test_pre_sr, test_post_sr, 
                                      "two-sided")
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_additional_vars_reg_one_var_test(self):
        test_sr = generate_data(100, distribution_type="cont")
        test_additional_var = generate_data(100, distribution_type="cont", index=test_sr.index).rename("var_1")
        control_sr = generate_data(100, distribution_type="cont")
        control_additional_var = generate_data(100, distribution_type="cont", index=control_sr.index).rename("var_1")
        p_value = additional_vars_regression_test(
            control_sr,
            [control_additional_var],
            test_sr,
            [test_additional_var],
            "two-sided"
        )
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_additional_vars_reg_one_var_test_fail(self):
        test_sr = generate_data(100, distribution_type="cont")
        test_additional_var = generate_data(100, distribution_type="cont", index=test_sr.index).rename("var_1")
        control_sr = generate_data(100, distribution_type="cont")
        control_additional_var = generate_data(100, distribution_type="cont", index=control_sr.index).rename("var_2")
        self.assertRaises(AssertionError, additional_vars_regression_test,
                          control_sr, [control_additional_var],
                          test_sr, [test_additional_var], "two-sided")

    def test_additional_vars_reg_multiple_vars_test(self):
        test_sr = generate_data(100, distribution_type="cont")
        test_additional_var1 = generate_data(100, distribution_type="cont", index=test_sr.index).rename("var_1")
        test_additional_var2 = generate_data(100, distribution_type="cont", index=test_sr.index).rename("var_1")
        control_sr = generate_data(100, distribution_type="cont")
        control_additional_var1 = generate_data(100, distribution_type="cont", index=control_sr.index).rename("var_1")
        control_additional_var2 = generate_data(100, distribution_type="cont", index=control_sr.index).rename("var_1")
        p_value = additional_vars_regression_test(
            control_sr,
            [control_additional_var1, control_additional_var2],
            test_sr,
            [test_additional_var1, test_additional_var2],
            "two-sided"
        )
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_additional_vars_reg_no_vars_test(self):
        test_sr = generate_data(100, distribution_type="cont")
        control_sr = generate_data(100, distribution_type="cont")
        self.assertRaises(AssertionError, additional_vars_regression_test, control_sr, [], test_sr, [], "two-sided")

    def test_cuped_ttest(self):
        test_sr = generate_data(100, distribution_type="cont")
        test_covariant_sr = generate_data(100, distribution_type="cont", index=test_sr.index).rename("var_1")
        control_sr = generate_data(100, distribution_type="cont")
        control_covariant_sr = generate_data(100, distribution_type="cont", index=control_sr.index).rename("var_1")
        p_value = cuped_ttest(
            control_sr,
            control_covariant_sr,
            test_sr,
            test_covariant_sr,
            "two-sided",
        )
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")
