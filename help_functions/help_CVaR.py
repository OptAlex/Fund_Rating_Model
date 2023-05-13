import pandas as pd
import numpy as np


def get_CVaR(df: pd.DataFrame, level: float) -> float:
    """
    Get the CVaR out of the pnl simulations.
    :param df: DataFrame with simulated pnl
    :param level: level to estimate the CVaR
    :return: CVaR for the specified percentile of the pnl.
    """
    pnl = df.sum(axis=0)
    CVaR = np.percentile(pnl, level)
    return CVaR

