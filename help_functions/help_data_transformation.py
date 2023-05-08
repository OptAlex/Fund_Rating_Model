# help functions related to data transformation
from const import *
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


def create_log_returns(str_raw_data_name, bool_drop_date=True):
    """
    Create log returns out of historical prices time series.
    :param str_raw_data_name: name of the Excel file
    :param bool_drop_date: if true, drop date column
    :return: dataframe with col = tickers, rows = days, values = log returns
    """

    df_raw_data = pd.read_excel(PATH_DATA + str_raw_data_name)
    df_data = input_data(df_raw_data)

    df_log_returns = np.log(df_data / df_data.shift(1))
    df_log_returns = df_log_returns[1:]

    if bool_drop_date:
        df_log_returns = df_log_returns.reset_index(drop=True)

    # Extract the last string from each column name and rename the column
    df_log_returns.rename(columns=lambda x: x.split()[-1], inplace=True)

    return df_log_returns


def get_fund_dict(df: pd.DataFrame) -> dict:
    """
    Create a dictionary of all funds with returns.
    :param df: df with the historical returns
    :return: dictionary with the fund name and the historical returns of the fund
    """
    fund_dict = {}
    for col in df.columns:
        returns = df[col].values
        fund_dict[col] = returns
    return fund_dict
