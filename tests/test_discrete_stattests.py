import unittest

import numpy as np
from abtoolkit.discrete.stattests import conversion_ztest
from abtoolkit.discrete.stattests import bayesian_test
from abtoolkit.utils import generate_data


class TestStatTests(unittest.TestCase):
    def test_ztest(self):
        test_sr = generate_data(100, distribution_type="disc")
        control_sr = generate_data(100, distribution_type="disc")
        p_value = conversion_ztest(
            control_sr.sum(),
            len(control_sr),
            test_sr.sum(),
            len(test_sr),
            "less"
        )
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_bayesian_test(self):
        test_sr = generate_data(100, distribution_type="disc")
        control_sr = generate_data(100, distribution_type="disc")
        p_value = bayesian_test(
            control_sr.sum(),
            len(control_sr),
            test_sr.sum(),
            len(test_sr),
            'less'
        )
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")

    def test_bayesian_test_prior(self):
        test_sr = generate_data(100, distribution_type="disc")
        control_sr = generate_data(100, distribution_type="disc")
        p_value = bayesian_test(
            control_sr.sum(),
            len(control_sr),
            test_sr.sum(),
            len(test_sr),
            'less',
            prior_positives_count=10,
            prior_negatives_count=10,
        )
        self.assertTrue(0 <= p_value <= 1, f"Wrong value for p-value: {p_value}")
