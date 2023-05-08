from help_functions.help_data_transformation import create_log_returns
from help_functions.help_functions_default_prob import calc_threshold_violation
import pandas as pd
from const import *


df_log_returns = create_log_returns("ETF_List.xlsx")

# Copula model
#
# df_sim_returns = pd.read_csv(File with simulated returns from R)
#
# df_all_violations = pd.DataFrame()
# for thold in [-100, -75, -50, -35, -25, -15, -10, -5]:
#   df_threshold_violations = calc_threshold_violation(df_sim_returns, threshold = thold)
#   df_all_violations = pd.concat[df_all_violations, df_threshold_violations]
