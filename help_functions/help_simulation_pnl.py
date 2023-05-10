import pandas as pd
import numpy as np
from help_functions.help_data_transformation import log_returns_to_normal_returns

def simulate_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate the PnL of a fund.
    :param df: DataFrame with log returns
    :return: DataFrame with the final cumulative returns after each simulation
    """
    # Get normal returns to calculate the portfolio return by using the weighted average
    normal_returns = log_returns_to_normal_returns(df)
    # calculate the mean of each row to get the weighted average return
    portfolio_return = normal_returns.mean(axis=1)
    cum_returns = np.cumprod(1 + portfolio_return)
    # substract -1 from all values as we add 1 when calculating the cumprod
    cum_returns = cum_returns - 1
    # create new DataFrame with last value of cum_returns
    pnl = pd.DataFrame({'Portfolio Returns': [cum_returns.iloc[-1]]})
    return pnl
