# Example file for creating help functions which can be used in the whole project
from const import *
import pandas as pd
import numpy as np

def input_data(df, date_column="Date"):
    """
    outputs correctly formatted data to perform operation.
    :param date_column: date column name
    :param df: df from source, i.e. excel
    :return: correctly formatted df
    """
    df[date_column] = pd.to_datetime(df[date_column], format="%Y-%M-%D")
    df_indexed = df.set_index(df[date_column])
    df_indexed = df_indexed.drop(columns=[date_column])
    df_dropped = df_indexed.dropna()

    return df_dropped

def log_returns(df):
    df_returns = df.sort_index(ascending=True).pct_change().dropna()
    df_log_returns = np.log(1+df_returns)

    # Extract the last string from each column name and rename the column
    df_log_returns.rename(columns=lambda x: "log_returns_" + x.split()[-1], inplace=True)

    return df_log_returns
