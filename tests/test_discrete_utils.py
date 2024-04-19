import unittest
from abtoolkit.discrete.utils import estimate_sample_size_by_mde
from abtoolkit.discrete.utils import estimate_mde_by_sample_size
from abtoolkit.discrete.utils import estimate_ci_binomial


class TestSampleSizeEstimation(unittest.TestCase):
    def test_calculate_sample_size_by_mde(self):
        sample_size = estimate_sample_size_by_mde(0.07, 0.05, 0.8, 0.03, alternative="two-sided")
        self.assertTrue(isinstance(sample_size, int), "Estimated sample size has wrong type")
        self.assertTrue(sample_size > 0, f"Estimated sample size has negative value: {sample_size}")

    def test_calculate_mde_by_sample_size(self):
        mde = estimate_mde_by_sample_size(0.07, 0.05, 0.8, 1136, alternative="two-sided")
        self.assertTrue(isinstance(mde, float), "Estimated MDE has wrong type")
        self.assertTrue(mde > 0, f"Estimated MDE is negative: {mde}")

    def test_estimate_ci_binomial(self):
        ci = estimate_ci_binomial(0.07, 1136, 0.5)
        self.assertTrue(len(ci) == 2, f"Got wrong confidence interval: {ci}")
        self.assertTrue(ci[0] > 0, f"Estimated ci has negative value: {ci[0]}")
        self.assertTrue(ci[1] > 0, f"Estimated ci has negative value: {ci[1]}")
