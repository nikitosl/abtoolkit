# ab-toolkit
Set of tools for AA and AB tests.

## Continuous variables analysis
#### Sample size estimation:
```
from src.continuous.sample_size_estimation import calculate_sample_size_by_mde
calculate_sample_size_by_mde(
    std=variable.std(),
    alpha=alpha_level, 
    power=power, 
    mde=mde
)
```

#### AA and AB tests simulation:
Using ```src.continuous.simulation.StatTestsSimulation``` class you can simulate and check different stat-test, 
compare them in terms of stat test power to choose the best test for your data. As result of simulation for each 
stat test you will get the 1-st Type error estimation with confidence interval, 2-nd Type error estimation with 
confidence interval and plot of p-value distribution for different tests.

```
from src.continuous.simulation import StatTestsSimulation
simulation = StatTestsSimulation(
    control,
    test,
    stattests_list=["ttest", "regression_test", "cuped_ttest", "did_regression_test", "additional_vars_regression_test"],
    experiments_num=experiments_num,
    sample_size=sample_size,
    mde=mde,
    alpha_level=alpha_level,

    control_previous_values=control_previous_value,
    test_previous_values=test_previous_value,
    control_cuped_covariant=control_previous_value,
    test_cuped_covariant=test_previous_value,
    control_additional_vars=[control_previous_value],
    test_additional_vars=[test_previous_value],
)
simulation.run()  # Run simulation
simulation.print_results()  # Print results of simulation
simulation.plot_p_values()  # Plot p-values distribution
```
Output:
![Screenshot 2024-03-20 at 14.55.06.png](..%2F..%2F..%2F..%2F..%2Fvar%2Ffolders%2Ftd%2F4g0swlbs5yg351q8l_8q7zt59mjw37%2FT%2FTemporaryItems%2FNSIRD_screencaptureui_mLjrT8%2FScreenshot%202024-03-20%20at%2014.55.06.png)
![Screenshot 2024-03-20 at 14.41.25.png](..%2F..%2F..%2F..%2F..%2Fvar%2Ffolders%2Ftd%2F4g0swlbs5yg351q8l_8q7zt59mjw37%2FT%2FTemporaryItems%2FNSIRD_screencaptureui_uC7yg9%2FScreenshot%202024-03-20%20at%2014.41.25.png)
Full example of usage you can find in ```examples/continuous_var_analysis.py``` script.

#### Next stat tests implemented for treatment effect estimation:
- ***T-Test*** - estimates treatment effect by comparing variables between test and control groups.
- ***Regression Test*** - estimates treatment effect using linear regression by tested predicting variable. 
Fact of treatment represented in model as binary flag (treated or not). Weight for this flag show significant 
of treatment impact.
```y = bias + w * treated```
- ***Regression Difference-in-Difference Test*** - estimates treatment effect using linear regression by predicting
difference between test and control groups whist represented as difference between current variable value and 
previous period variable value (two differences). Weight for treated and current variable values shows 
significant of treatment. ```y = bias + w0 * treated + w1 * after + w2 * treated * after```
- ***CUPED*** - estimates treatment effect by comparing variables between test and control groups and uses covariant 
to reduce variance and speedup test. ```y = y - Q * covariant```, where ```Q = cov(y, covariant) / var(covariant)```. 
Cuped variable has same mean value (unbiased), but smaller variance, that speedup test.
- ***Regression with Additional Variables*** - estimates treatment effect using linear regression by predicting 
tested variable with additional variables, which describe part of main variable variance and speedup test. 
Fact of treatment represented in model as binary flag (treated or not). Weight for this flag show significant 
of treatment impact.
```y = bias + w0 * treated + w1 * additional_variable1 + w2 * additional_variable2 + ...```


## Discrete variables analysis
#### Sample size estimation:
```
from src.discrete.sample_size_estimation import estimate_ci_binomial
estimate_ci_binomial(
    p, 
    sample_size, 
    alpha=0.05
)
```
#### AA and AB tests simulation:
ToBeDone
#### Next stat tests implemented for treatment effect estimation:
ToBeDone

You can find examples of toolkit usage in ```examples/``` directory.