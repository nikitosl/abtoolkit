from abtoolkit.continuous.simulation import StatTestsSimulation
from abtoolkit.continuous.utils import estimate_sample_size_by_mde
from abtoolkit.utils import generate_data


if __name__ == '__main__':
    # Fix global params
    mde = 2
    alpha_level = 0.05
    power = 0.8
    examples_num = 2000  # Number of examples in test and control group
    experiments_num = 200  # Number of experiments to run for each stattest
    alternative = 'two-sided'

    # Generate variable
    variable = generate_data(examples_num, distribution_type="cont")
    # Generate previous value of variable (will be used to reduce variance) and speedup tests
    previous_value = generate_data(examples_num, distribution_type="cont", index=variable.index).rename("prev")

    # Estimate sample_size need for test
    sample_size = estimate_sample_size_by_mde(
        std=variable.std(),
        alpha=alpha_level, power=power, mde=mde, alternative=alternative)
    print(f"Minimum sample size for each group is {sample_size}")

    sim = StatTestsSimulation(
        variable,
        stattests_list=["ttest", "diff_ttest", "regression_test", "cuped_ttest", "did_regression_test",
                        "additional_vars_regression_test"],
        alternative=alternative,
        experiments_num=experiments_num,  # Run each stattest 10 times
        sample_size=sample_size,  # Take 50 samples from variables
        mde=mde,  # Trying to detect this effect (very big for our simulated data)
        alpha_level=alpha_level,  # Fix alpha level on 5%

        previous_values=previous_value,
        cuped_covariant=previous_value,
        additional_vars=[previous_value],
    )
    info = sim.run()  # Get dictionary with information about tests
    sim.print_results()  # Print results of simulation
    sim.plot_p_values()  # Plot p-values distribution
