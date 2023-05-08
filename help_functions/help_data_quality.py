# help functions related to data quality


def check_zero_returns(df):
    """
    Check the number of zeros per column in a dataframe and return the percentage of zeros. Bloomberg fills missing
    historical prices with the last reported price. This corresponds to a zero return. To ensure, that there is only a
    small amount of autofilled prices in the data, this function can be used.

    :param df: dataframe of returns
    :return: percentage of zeros expressed in decimals
    """
    zero_counts = df.eq(0).sum()
    zero_ratios = zero_counts / len(df)

    return zero_ratios
