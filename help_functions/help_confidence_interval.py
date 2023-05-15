import numpy as np
import pandas as pd
from scipy import stats


def get_confidence_interval(data, level):
    """
    Calculate the confidence interval of a sample assuming a t-distribution.
    :param data: A 1D array of sample data
    :param level: The confidence level in percentage (e.g. 95 for 95% confidence interval)
    :return: A tuple (lower_ci, upper_ci) representing the lower and upper bounds of the confidence interval.
    """
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation
    t_value = stats.t.ppf(1 - (1 - level / 100) / 2, n - 1)
    margin_of_error = t_value * sample_std / np.sqrt(n)
    lower_ci = sample_mean - margin_of_error
    upper_ci = sample_mean + margin_of_error
    return lower_ci, upper_ci


def confidence_interval_no_distr(data, alpha):
    df_all_bounds = pd.DataFrame(index=["lower", "upper"])

    for level in alpha:
        col = f"CI_{level * 100}%"
        # Calculate the statistic of interest for each bootstrap sample
        data_means = np.mean(data, axis=1)

        # Calculate the confidence interval
        lower = np.percentile(data_means, (1 - level) / 2 * 100)
        upper = np.percentile(data_means, (1 + level) / 2 * 100)

        df_all_bounds.loc["lower", col] = lower
        df_all_bounds.loc["upper", col] = upper

    return df_all_bounds
