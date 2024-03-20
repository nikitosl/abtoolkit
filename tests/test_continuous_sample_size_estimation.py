import unittest

import numpy as np
import pandas as pd

from src.continuous.sample_size_estimation import calculate_mde_by_sample_size
from src.continuous.sample_size_estimation import calculate_sample_size_by_mde


class TestSampleSizeEstimation(unittest.TestCase):
    def test_calculate_sample_size_by_mde(self):
        true_sample_size = 36
        sample_size = calculate_sample_size_by_mde(3, 0.05, 0.8, 2)
        self.assertEquals(true_sample_size, sample_size, "Estimated wrong sample size")

    def test_calculate_mde_by_sample_size(self):
        true_mde = 2
        mde = calculate_mde_by_sample_size(3, 0.05, 0.8, 36)
        self.assertEquals(true_mde, round(mde), "Estimated wrong MDE")
