from help_functions.help_data_transformation import input_data, log_returns
import pandas as pd
from const import *


# generate log return df for R
excel_data = pd.read_excel(PATH_DATA + "Second Fund list.xlsx")
data = input_data(excel_data)
log_returns = log_returns(data)
log_returns.to_csv(PATH_DATA+"log_returns.csv")