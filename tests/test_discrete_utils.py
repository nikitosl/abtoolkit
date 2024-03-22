import unittest
from abtoolkit.discrete.utils import calculate_sample_size
from abtoolkit.discrete.utils import estimate_ci_binomial


class TestSampleSizeEstimation(unittest.TestCase):
    def test_calculate_sample_size_by_mde(self):
        sample_size = calculate_sample_size(0.07, 0.05, 0.8, 0.03)
        self.assertTrue(isinstance(sample_size, int), "Estimated sample size has wrong type")
        self.assertTrue(sample_size > 0, f"Estimated sample size has negative value: {sample_size}")

    def test_estimate_ci_binomial(self):
        ci = estimate_ci_binomial(0.07, 1136, 0.5)
        self.assertTrue(len(ci) == 2, f"Got wrong confidence interval: {ci}")
        self.assertTrue(ci[0] > 0, f"Estimated ci has negative value: {ci[0]}")
        self.assertTrue(ci[1] > 0, f"Estimated ci has negative value: {ci[1]}")
