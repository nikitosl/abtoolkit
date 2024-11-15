import numpy as np

from abtoolkit.discrete.simulation import StatTestsSimulation
from abtoolkit.discrete.utils import estimate_sample_size_by_mde
from abtoolkit.utils import generate_data


if __name__ == '__main__':
    # Fix global params
    mde = 0.05
    alpha_level = 0.05
    power = 0.8
    examples_num = 2000  # Number of examples in test and control group
    experiments_num = 1000  # Number of experiments to run for each stattest
    alternative = 'less'

    # Generate variable
    variable = generate_data(examples_num, distribution_type="disc")

    proba = variable.sum() / variable.count()

    # Estimate sample_size need for test
    sample_size = estimate_sample_size_by_mde(p=proba, alpha=alpha_level, power=power, mde=mde, alternative=alternative)
    print(f"Minimum sample size for each group is {sample_size}")

    sim = StatTestsSimulation(
        count=variable.sum(),
        objects_num=variable.count(),
        stattests_list=["conversion_ztest", "bayesian_test"],
        alternative=alternative,
        experiments_num=experiments_num,  # Run each stattest 10 times
        sample_size=sample_size,  # Take 50 samples from variables
        mde=mde,  # Trying to detect this effect (very big for our simulated data)
        alpha_level=alpha_level,  # Fix alpha level on 5%
        power=power,
        bayesian_prior_positives=1,
        bayesian_prior_negatives=1,
    )
    info = sim.run()  # Get dictionary with information about tests
    sim.print_results()  # Print results of simulation
    sim.plot_p_values()  # Plot p-values distribution
