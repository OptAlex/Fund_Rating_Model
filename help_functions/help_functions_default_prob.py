import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm


def get_fund_dict(df: pd.DataFrame) -> dict:
    """
    Create a dictionary of all funds with price & returns.
    :param df: df with the historical prices and returns
    :return: dictionary with the fund name and the historical prices and returns of the fund
    """
    fund_dict = {}
    for col in df.columns:
        if col.startswith("price_"):
            fund_name = col.split("_")[1]
            prices = df[col].values
            returns = df[f"return_{fund_name}"].values
            fund_dict[fund_name] = (prices, returns)
    return fund_dict


def simulate_returns(garch_parameters: list, num_days_to_simulate: int) -> list:
    """
    Function to simulate the future returns
    :param garch_parameters: alpha, beta, sigma
    :param num_days_to_simulate: number of days on which the simulation is performed
    :return: simulated returns for the number of prediction days
    """
    sim_mod = arch_model(None, p=1, q=1, rescale="False")
    sim_data = sim_mod.simulate(garch_parameters, num_days_to_simulate)
    return sim_data["data"]


def run_garch_model(returns: list) -> list:
    """
    Simple function to run a GARCH model
    :param returns: returns on which to run the model
    :return: the residuals of the model
    """
    # using GARCH(1,1)
    am = arch_model(returns, p=1, q=1, rescale="False")
    res = am.fit(disp="off")
    return res

def get_ci_from_df(df: pd.DataFrame, confidence_level: float) -> dict:
    """
    Calculate the mean and confidence interval for the portfolio based on a DataFrame with default probabilities.
    :param df: DataFrame with default probabilities for the portfolio.
    :param confidence_level: Desired confidence level.
    :return: Dictionary containing the mean and confidence interval for the portfolio.
    """
    n = len(df)
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    mean_dict = {}
    ci_dict = {}

    for column in df.columns:
        data = df[column].to_numpy()
        mean = np.mean(data)
        std_error = np.std(data, ddof=1) / np.sqrt(n)
        margin_of_error = z_score * std_error
        lower_ci = mean - margin_of_error
        upper_ci = mean + margin_of_error

        mean_dict[column] = mean
        ci_dict[column] = (lower_ci, upper_ci)

    return {'mean': mean_dict, 'confidence_interval': ci_dict}
