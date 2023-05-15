import pandas as pd
import numpy as np


def get_CVaR(df, level, stop_loss, threshold):
    """
    Get the CVaR out of the pnl simulations.
    :param df: DataFrame with simulated pnl
    :param level: Level to estimate the CVaR
    :param stop_loss: Defines a stop loss for the portfolio
    :param threshold: Defines the threshold of the stop loss
    :return: CVaR for the specified percentile of the pnl.
    """
    cumulative_returns = np.cumprod(df + 1, axis=0) - 1
    if stop_loss == True:
        final_returns = np.where(cumulative_returns <= -threshold, -threshold, cumulative_returns)
        final_returns = final_returns[-1, :]
    else:
        final_returns = cumulative_returns[-1, :]
    CVaR = np.percentile(a=final_returns, q=level)
    return CVaR

