# help functions related to data transformation
# from const import *
import pandas as pd
import numpy as np


def input_data(df, date_column="Date"):
    """
    Outputs correctly formatted data to perform operations.
    :param date_column: date column name
    :param df: df from source, i.e. excel
    :return: correctly formatted df
    """
    df[date_column] = pd.to_datetime(df[date_column], format="%Y-%M-%D")
    df_indexed = df.set_index(df[date_column])
    df_indexed = df_indexed.drop(columns=[date_column])
    df_sorted = df_indexed.sort_index(ascending=True)
    df_dropped = df_sorted.dropna()

    return df_dropped


def create_log_returns(path_raw_data_name, bool_drop_date=True):
    """
    Create log returns out of historical prices time series.
    :param str_raw_data_name: name of the Excel file
    :param bool_drop_date: if true, drop date column
    :return: dataframe with col = tickers, rows = days, values = log returns
    """

    df_raw_data = pd.read_excel(path_raw_data_name)
    df_data = input_data(df_raw_data)

    df_log_returns = np.log(df_data).diff()
    df_log_returns = df_log_returns[1:]

    if bool_drop_date:
        df_log_returns = df_log_returns.reset_index(drop=True)

    # Extract the last string from each column name and rename the column
    df_log_returns.rename(columns=lambda x: x.split()[-1], inplace=True)

    return df_log_returns

def get_fund_dict(df: pd.DataFrame) -> dict:
    """
    Create a dictionary of all funds with returns.
    :param df: df with the log returns
    :return: dictionary with the fund name and the log returns of the fund
    """
    fund_dict = {}

    for col in df.columns:
        returns = df[col].values
        fund_dict[col] = returns

    return fund_dict


def convert_returns(df, bool_to_log=True):
    """
    Convert returns between standard returns and log returns.
    :param df: dataframe with returns
    :param bool_to_log: True, if convert to log returns
    :return: df with converted returns
    """
    if bool_to_log:
        # Convert standard returns to log returns
        if (df < -1).any().any():
            raise ValueError("Cannot convert negative returns to log returns.")
        return np.log(1 + df)

    # Convert log returns to standard returns
    return np.exp(df) - 1


def calculate_portfolio_return(df_returns, weights=None):
    # If weights are not provided, use equal weights
    if weights is None:
        weights = np.ones(len(df_returns.columns)) / len(df_returns.columns)

    # Calculate the portfolio return as the weighted average of asset returns
    df_portfolio_return = (df_returns * weights).sum(axis=1)

    return df_portfolio_return
