# ABToolkit
Set of tools for AA and AB tests, sample size estimation, confidence intervals estimation. 
For continuous and discrete variables.

## Install using pip:
```pip install abtoolkit```

## Continuous variables analysis
#### Sample size estimation:
```
from abtoolkit.continuous.utils import calculate_sample_size_by_mde
calculate_sample_size_by_mde(
    std=variable.std(),
    alpha=alpha_level, 
    power=power, 
    mde=mde
)
```

#### AA and AB tests simulation:
Using ```abtoolkit.continuous.simulation.StatTestsSimulation``` class you can simulate and check different stat-test, 
compare them in terms of stat test power to choose the best test for your data. As result of simulation for each 
stat test you will get the 1-st Type error estimation with confidence interval, 2-nd Type error estimation with 
confidence interval and plot of p-value distribution for different tests.

```
from abtoolkit.continuous.simulation import StatTestsSimulation
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
![output-plot.png](static%2Foutput-plot.png)
![p-value-plot.png](static%2Fp-value-plot.png)

Full example of usage you can find in ```examples/continuous_var_analysis.py``` script.

#### Next stat tests implemented for treatment effect estimation:
- ***T-Test*** - estimates treatment effect by comparing variables between test and control groups.
- ***Difference T-Test*** - estimates treatment effect by comparing difference between actual and previous values 
of variables in test and control groups.
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
from abtoolkit.discrete.utils import estimate_ci_binomial
estimate_ci_binomial(
    p, 
    sample_size, 
    alpha=0.05
)
```
#### AA and AB tests simulation:
To Be Done
#### Next stat tests implemented for treatment effect estimation:
To Be Done

---
You can find examples of toolkit usage in [examples/](https://github.com/nikitosl/abtoolkit/tree/master/examples) directory.