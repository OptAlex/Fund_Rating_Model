# help functions to calculate the default probabilities

import pandas as pd
import numpy as np
from help_functions.help_data_transformation import get_fund_dict
from const import THRESHOLDS


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


def calc_default_probs(dfs, bool_indiv=False):
    """
    Calculate default probabilities based on threshold violations
    :param dfs: dataframe with simulations
    :param bool_indiv: individual default probabilities if True
    :return: default probabilities
    """
    # Default prob of individual funds
    df_all_defaults = pd.DataFrame()
    for threshold in THRESHOLDS:
        for i, df in enumerate(dfs):
            sim_results = df

            df_default = calc_threshold_violation(df=sim_results, threshold=threshold)
            str_threshold = f"Threshold_{round(threshold * 100, 2)}%"
            if threshold < 0.1:
                str_threshold = f"Threshold_0{round(threshold * 100, 2)}%"
            df_default.index = [str_threshold]
            df_all_defaults = pd.concat([df_all_defaults, df_default])

    if bool_indiv:
        df_all_def_prob = df_all_defaults.groupby(
            df_all_defaults.index
        ).mean()  # default prob
        df_all_def_prob_sorted = df_all_def_prob.sort_index(axis=0)

    else:
        df_all_def_prob = df_all_defaults.sum(axis=1) / df_all_defaults.shape[1]
        df_all_def_prob_sorted = df_all_def_prob.sort_index(axis=0)

    return df_all_def_prob_sorted
