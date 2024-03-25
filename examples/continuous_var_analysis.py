import numpy as np

from abtoolkit.continuous.simulation import StatTestsSimulation
from abtoolkit.continuous.utils import estimate_sample_size_by_mde
from abtoolkit.continuous.utils import generate_data


if __name__ == '__main__':
    # Fix global params
    mde = 2
    alpha_level = 0.05
    power = 0.8
    examples_num = 1000  # Number of examples in test and control group
    experiments_num = 1000  # Number of experiments to run for each stattest
    alternative = 'two-sided'

    # Generate test variable
    test_sr = generate_data(examples_num)
    # Generate previous value of test variable (will be used to reduce variance) and speedup tests
    test_previous_value = generate_data(examples_num, index=test_sr.index).rename("prev")

    # Generate control variable
    control_sr = generate_data(examples_num)
    # Generate previous value of control variable (will be used to reduce variance) and speedup tests
    control_previous_value = generate_data(examples_num, index=control_sr.index).rename("prev")

    # Estimate sample_size need for test
    sample_size = estimate_sample_size_by_mde(
        std=np.concatenate([control_sr, test_sr], axis=0).std(),
        alpha=alpha_level, power=power, mde=mde, alternative=alternative)
    print(f"Minimum sample size for each group is {sample_size}")

    sim = StatTestsSimulation(
        control_sr,
        test_sr,
        stattests_list=["ttest", "diff_ttest", "regression_test", "cuped_ttest", "did_regression_test",
                        "additional_vars_regression_test"],
        alternative=alternative,
        experiments_num=experiments_num,  # Run each stattest 10 times
        sample_size=sample_size,  # Take 50 samples from variables
        mde=mde,  # Trying to detect this effect (very big for our simulated data)
        alpha_level=alpha_level,  # Fix alpha level on 5%

        control_previous_values=control_previous_value,
        test_previous_values=test_previous_value,
        control_cuped_covariant=control_previous_value,
        test_cuped_covariant=test_previous_value,
        control_additional_vars=[control_previous_value],
        test_additional_vars=[test_previous_value],
    )
    info = sim.run()  # Get dictionary with information about tests
    sim.print_results()  # Print results of simulation
    sim.plot_p_values()  # Plot p-values distribution
