import numpy as np
from help_functions.help_confidence_interval import get_confidence_interval


def get_CVaR(df, level, stop_loss, threshold, confidence_level):
    """
    Get the CVaR out of the pnl simulations.
    :param df: DataFrame with simulated pnl
    :param level: Level to estimate the CVaR
    :param stop_loss: Defines a stop loss for the portfolio
    :param threshold: Defines the threshold of the stop loss
    :param confidence_level: The confidence level in percentage (default to 95)
    :return: A tuple (CVaR, (lower_ci, upper_ci)) representing the CVaR value and its confidence interval.
    """
    cumulative_returns = np.cumprod(df + 1, axis=0) - 1
    if stop_loss == True:
        final_returns = np.where(cumulative_returns <= -threshold, -threshold, cumulative_returns)
        final_returns = final_returns[-1, :]
    else:
        final_returns = cumulative_returns[-1, :]
    CVaR = np.percentile(a=final_returns, q=level)
    lower_ci, upper_ci = get_confidence_interval(final_returns, confidence_level)

    return CVaR, (lower_ci, upper_ci)

