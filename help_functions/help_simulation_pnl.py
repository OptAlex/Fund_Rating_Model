import pandas as pd
import numpy as np

def simulate_pnl(df: pd.DataFrame, interest: str) -> pd.DataFrame:
    """
    Simulate the PnL of a fund.
    :param df: DataFrame with the historical returns
    :return: DataFrame with the final cumulative returns after each simulation
    """
    fund_dict = get_fund_dict(df)
    dict_simulations = {fund_name: [] for fund_name in fund_dict.keys()}

    if interest == "Portfolio":
        normal_returns = log_returns_to_normal_returns(df)
        #get weighted avg
        cum_returns = np.cumprod(1 + returns)
        dict_simulations[fund_name].append(cum_returns[-1])
    else:
        for fund_name, returns in fund_dict.items():
            cum_returns = np.cumsum(returns)
            dict_simulations[fund_name].append(cum_returns[-1])

    df_pnl = pd.DataFrame(dict_simulations)
    return pd.DataFrame(df_pnl)
