# help functions related to CVaR
import numpy as np
import pandas as pd


def get_CVaR(df, level, stop_loss=False, threshold=None):
    """
    Get the CVaR out of the pnl simulations.
    :param df: DataFrame with simulated pnl
    :param level: Level to estimate the CVaR
    :param stop_loss: Defines a stop loss for the portfolio
    :param threshold: Defines the threshold of the stop loss
    :return: CVaR
    """
    final_returns = pd.DataFrame()
    cumulative_returns = np.cumprod(df + 1, axis=0) - 1
    if stop_loss:
        for column in cumulative_returns.columns:
            if (cumulative_returns[column] < -threshold).any():
                first_occurrence = cumulative_returns.index[
                    cumulative_returns[column] < -threshold
                ][0]
                cumulative_returns.loc[first_occurrence:, column] = -threshold

            final_returns = cumulative_returns.iloc[-1]
    else:
        final_returns = cumulative_returns.iloc[-1]

    CVaR = np.percentile(a=final_returns, q=level)

    return CVaR
