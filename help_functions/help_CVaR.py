import pandas as pd
import numpy as np


def get_CVaR(df, level):
    """
    Get the CVaR out of the pnl simulations.
    :param df: DataFrame with simulated pnl
    :param level: level to estimate the CVaR
    :return: CVaR for the specified percentile of the pnl.
    """
    # pnl = df.sum(axis=0)
    cumulative_returns = np.cumprod(df+1) - 1
    CVaR = np.percentile(a=cumulative_returns, q=level)
    return CVaR
