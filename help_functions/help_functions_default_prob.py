# help functions to calculate the default probabilities

import pandas as pd
import numpy as np
from help_functions.help_data_transformation import get_fund_dict


def calc_threshold_violation(df, threshold):
    """
    Calculate if a fund violates the threshold.
    :param df: df with returns
    :param threshold: threshold which defines the default of a fund; default if <; i.e. 0.1 for 10 percent loss
    :return: df with 1, if violated, 0 else
    """
    fund_dict = get_fund_dict(df)
    dict_violations = {fund_name: [] for fund_name in fund_dict.keys()}

    for fund_name, returns in fund_dict.items():
        arr_cumulative_returns = np.cumsum(returns)

        # 1 if threshold strictly violated, 0 else
        binary_violations = np.sum(np.any(arr_cumulative_returns < -threshold))
        dict_violations[fund_name].append(binary_violations)

    df_defaults = pd.DataFrame(dict_violations)

    return df_defaults
