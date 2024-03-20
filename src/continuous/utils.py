from scipy import stats
import pandas as pd
import numpy as np


def generate_data_for_regression_test(size, add_index=True, index=None):
    variable = np.random.normal(0, 3, size=size)
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
