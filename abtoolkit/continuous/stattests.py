from typing import List

import linearmodels as lm
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def ttest(
        control: pd.Series,
        test: pd.Series,
) -> float:
    """
    Simple two-side t-test
    :param control: pd.Series for control sample
    :param test: pd.Series for test sample
    :return: p-value
    """

    return ttest_ind(control, test, alternative="less").pvalue


def difference_ttest(
        control: pd.Series,
        control_pre: pd.Series,
        test: pd.Series,
        test_pre: pd.Series,
) -> float:
    """
    Estimation treatment effect using ttest and CUPED to increase test's power
    :param control: pd.Series, control sample
    :param control_pre: pd.Series, control previous period value
    :param test: pd.Series, test sample
    :param test_pre: pd.Series, test previous period value
    :return: p-value
    """
    control = control - control_pre
    test = test - test_pre

    return ttest(control, test)


def cuped_ttest(
        control: pd.Series,
        control_covariant: pd.Series,
        test: pd.Series,
        test_covariant: pd.Series,
) -> float:
    """
    Estimation treatment effect using ttest and CUPED to increase test's power
    :param control: pd.Series, control sample
    :param control_covariant: pd.Series, control sample covariant
    :param test: pd.Series, test sample
    :param test_covariant: pd.Series, test sample covariant
    :return: p-value
    """

    full_value = pd.concat([
        control.rename("value"),
        test.rename("value"),
    ], axis=0)

    full_covariant = pd.concat([
        control_covariant.rename("covariant"),
        test_covariant.rename("covariant"),
    ], axis=0)

    cov = np.cov(full_covariant, full_value)[0, 1]
    var = full_covariant.var()
    theta = cov / var

    cuped_test = test - theta * test_covariant
    cuped_control = control - theta * control_covariant

    return ttest(cuped_control, cuped_test)



def regression_test(
        control: pd.Series,
        test: pd.Series,
) -> float:
    """
    Treatment effect estimation using linear regression
    :param control: pd.Series with index [entity, dt], where dt could be int of datetime. Control sample
    :param test: pd.Series with index [entity, dt], where dt could be int of datetime. Test sample
    :return: p-value
    """
    df = pd.concat([
        control.rename("value").to_frame().assign(treated=0),
        test.rename("value").to_frame().assign(treated=1),
    ], axis=0)
    df["bias"] = 1

    if not isinstance(df.index, pd.MultiIndex):
        df["index1"] = 0
        df["index2"] = 1
        df = df.set_index(["index1", "index2"])

    mod = lm.PanelOLS.from_formula("value ~ bias + treated", data=df)
    result = mod.fit()
    return result.pvalues["treated"]


def did_regression_test(
        control: pd.Series,
        control_pre: pd.Series,
        test: pd.Series,
        test_pre: pd.Series,
) -> float:
    """
    Difference-in-Difference treatment effect estimation using linear regression.
    Calculates difference between current and last values in test and control groups and then
    calculates difference between differences to increase test power
    :param control_pre: pd.Series with index [entity, dt], where dt could be int of datetime.
    Control sample before treatment
    :param control: pd.Series with index [entity, dt], where dt could be int of datetime.
    Control sample after treatment
    :param test_pre: pd.Series with index [entity, dt], where dt could be int of datetime. Test sample before treatment
    :param test: pd.Series with index [entity, dt], where dt could be int of datetime. Test sample after treatment
    :return: p-value
    """
    df = pd.concat([
        control_pre.rename("value").to_frame().assign(treated=0).assign(after=0),
        control.rename("value").to_frame().assign(treated=0).assign(after=1),
        test_pre.rename("value").to_frame().assign(treated=1).assign(after=0),
        test.rename("value").to_frame().assign(treated=1).assign(after=1),
    ], axis=0)

    df["bias"] = 1

    if not isinstance(df.index, pd.MultiIndex):
        df["index1"] = 0
        df["index2"] = 1
        df = df.set_index(["index1", "index2"])

    mod = lm.PanelOLS.from_formula("value ~ bias + + after + treated + treated*after", data=df)
    result = mod.fit()
    return result.pvalues["treated:after"]


def additional_vars_regression_test(
        control: pd.Series,
        control_additional_vars: List[pd.Series],
        test: pd.Series,
        test_additional_vars: List[pd.Series],
) -> float:
    """
    Treatment effect estimation using additional variables in linear regression. Additional
    variables should reduce deviation of target variable and increase test power
    :param control: pd.Series with index [entity, dt], where dt could be int of datetime.
    Control sample
    :param control_additional_vars: List of pd.Series with index [entity, dt], where dt could be int of datetime.
    Additional variables which can describe some deviation of tested variable
    :param test: pd.Series with index [entity, dt], where dt could be int of datetime.
    Test sample
    :param test_additional_vars: List of pd.Series with index [entity, dt], where dt could be int of datetime.
    Additional variables which can describe some deviation of tested variable
    :return: p-value
    """

    assert len(test_additional_vars) > 0, "No additional vars for 'additional_vars_regression_test' test given"

    additional_vars_names_test = [v.name for v in test_additional_vars]
    additional_vars_names_control = [v.name for v in control_additional_vars]
    assert set(additional_vars_names_test) == set(additional_vars_names_control), \
        (f"Lists of control and test additional vars should the same. "
         f"Got {set(additional_vars_names_test)} vars for test "
         f"and {set(additional_vars_names_control)} vars for control")

    control_df = pd.concat([control.rename("value").to_frame()] + control_additional_vars, axis=1)
    test_df = pd.concat([test.rename("value").to_frame()] + test_additional_vars, axis=1)

    control_df.index = test_df.index
    df = pd.concat([
        control_df.assign(treated=0),
        test_df.assign(treated=1),
    ], axis=0)

    df["bias"] = 1

    if not isinstance(df.index, pd.MultiIndex):
        df["index1"] = 0
        df["index2"] = 1
        df = df.set_index(["index1", "index2"])

    additional_vars_formula = " + ".join(map(str, additional_vars_names_test))

    formula = f"value ~ bias + treated + {additional_vars_formula}"
    mod = lm.PanelOLS.from_formula(formula, data=df)
    result = mod.fit()
    return result.pvalues["treated"]
