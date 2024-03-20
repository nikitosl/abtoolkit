import unittest

from src.continuous.stattests import regression_test
from src.continuous.stattests import did_regression_test
from src.continuous.stattests import additional_vars_regression_test
from src.continuous.stattests import cuped_ttest
from src.continuous.stattests import difference_ttest
from src.continuous.utils import generate_data_for_regression_test


class TestStatTests(unittest.TestCase):
    def test_regression_test(self):
        test_sr = generate_data_for_regression_test(100)
        control_sr = generate_data_for_regression_test(100)
        p_value = regression_test(control_sr, test_sr)
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_diff_ttest(self):
        test_sr = generate_data_for_regression_test(100)
        test_pre_sr = generate_data_for_regression_test(100, index=test_sr.index).rename("var_1")
        control_sr = generate_data_for_regression_test(100)
        control_pre_sr = generate_data_for_regression_test(100, index=control_sr.index).rename("var_1")
        p_value = difference_ttest(control_pre_sr, control_sr, test_pre_sr, test_sr)
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_did_regression_test(self):
        test_pre_sr = generate_data_for_regression_test(100)
        test_post_sr = generate_data_for_regression_test(100)
        control_pre_sr = generate_data_for_regression_test(100)
        control_post_sr = generate_data_for_regression_test(100)
        p_value = did_regression_test(control_pre_sr, control_post_sr, test_pre_sr, test_post_sr)
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_additional_vars_reg_one_var_test(self):
        test_sr = generate_data_for_regression_test(100)
        test_additional_var = generate_data_for_regression_test(100, index=test_sr.index).rename("var_1")
        control_sr = generate_data_for_regression_test(100)
        control_additional_var = generate_data_for_regression_test(100, index=control_sr.index).rename("var_1")
        p_value = additional_vars_regression_test(
            control_sr,
            [control_additional_var],
            test_sr,
            [test_additional_var]
        )
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_additional_vars_reg_one_var_test_fail(self):
        test_sr = generate_data_for_regression_test(100)
        test_additional_var = generate_data_for_regression_test(100, index=test_sr.index).rename("var_1")
        control_sr = generate_data_for_regression_test(100)
        control_additional_var = generate_data_for_regression_test(100, index=control_sr.index).rename("var_2")
        self.assertRaises(AssertionError, additional_vars_regression_test,
                          control_sr, [control_additional_var],
                          test_sr, [test_additional_var])

    def test_additional_vars_reg_multiple_vars_test(self):
        test_sr = generate_data_for_regression_test(100)
        test_additional_var1 = generate_data_for_regression_test(100, index=test_sr.index).rename("var_1")
        test_additional_var2 = generate_data_for_regression_test(100, index=test_sr.index).rename("var_1")
        control_sr = generate_data_for_regression_test(100)
        control_additional_var1 = generate_data_for_regression_test(100, index=control_sr.index).rename("var_1")
        control_additional_var2 = generate_data_for_regression_test(100, index=control_sr.index).rename("var_1")
        p_value = additional_vars_regression_test(
            control_sr,
            [control_additional_var1, control_additional_var2],
            test_sr,
            [test_additional_var1, test_additional_var2]
        )
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_additional_vars_reg_no_vars_test(self):
        test_sr = generate_data_for_regression_test(100)
        control_sr = generate_data_for_regression_test(100)
        self.assertRaises(AssertionError, additional_vars_regression_test, control_sr, [], test_sr, [])

    def test_cuped_ttest(self):
        test_sr = generate_data_for_regression_test(100)
        test_covariant_sr = generate_data_for_regression_test(100, index=test_sr.index).rename("var_1")
        control_sr = generate_data_for_regression_test(100)
        control_covariant_sr = generate_data_for_regression_test(100, index=control_sr.index).rename("var_1")
        p_value = cuped_ttest(
            control_sr,
            control_covariant_sr,
            test_sr,
            test_covariant_sr,
        )
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")
