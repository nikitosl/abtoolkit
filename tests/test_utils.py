import unittest
import numpy as np
from abtoolkit.utils import check_clt


class TestUtils(unittest.TestCase):
    def test_check_central_limit_theorem_normal(self):
        var = np.random.normal(0, 2, size=10000)
        p_value = check_clt(var, do_plot_distribution=False, sample_size=10)
        self.assertTrue(p_value >= 0, f"p-value: {p_value}")

    def test_check_central_limit_theorem_continuous(self):
        var = np.random.chisquare(df=2, size=10000)
        p_value = check_clt(var, do_plot_distribution=False, sample_size=10000)
        self.assertTrue(p_value >= 0, f"p-value: {p_value}")

    def test_check_central_limit_theorem_discrete(self):
        var = np.random.choice([0, 1], p=[0.93, 0.07], size=10000)
        p_value = check_clt(var, do_plot_distribution=False, sample_size=50000)
        self.assertTrue(p_value >= 0, f"p-value: {p_value}")

    def test_check_central_limit_theorem_normal_median(self):
        var = np.random.normal(0, 2, size=10000)
        p_value = check_clt(var, do_plot_distribution=False, sample_size=10, metric_f=np.median)
        self.assertTrue(p_value >= 0, f"p-value: {p_value}")

    def test_check_central_limit_theorem_normal_percentile(self):
        var = np.random.normal(0, 2, size=10000)
        p_value = check_clt(var, do_plot_distribution=False, metric_f=np.percentile, q=0.9)
        self.assertTrue(p_value >= 0, f"p-value: {p_value}")
