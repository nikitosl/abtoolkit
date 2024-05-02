from abtoolkit.utils import check_clt
import numpy as np


if __name__ == '__main__':
    print(f"Checking Central Limit Theorem for Chi-Squared distributed variable")
    chi_variable = np.random.chisquare(df=1, size=10000)
    for sample_size in [10, 100, 1000, 5000]:
        pvalue = check_clt(chi_variable, sample_size=sample_size)
        print(f"\nSample_size = {sample_size}", end='\t')
        if pvalue < 0.05:
            print("\033[91m" + f"Fail, p-value={pvalue}" + "\033[0m")
        else:
            print("\033[92m" + f"Pass, p-value={pvalue}" + "\033[0m")

    print(f"Checking Central Limit Theorem for Normal distributed variable")
    normal_variable = np.random.normal(loc=0, scale=1, size=10000)
    for sample_size in [2, 10, 100]:
        pvalue = check_clt(normal_variable, sample_size=sample_size)
        print(f"\nSample_size = {sample_size}", end='\t')
        if pvalue < 0.05:
            print("\033[91m" + f"Fail, p-value={pvalue}" + "\033[0m")
        else:
            print("\033[92m" + f"Pass, p-value={pvalue}" + "\033[0m")
