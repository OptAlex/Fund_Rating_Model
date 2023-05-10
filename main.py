#from help_functions.help_data_transformation import create_log_returns
#from help_functions.help_functions_default_prob import calc_threshold_violation
import pandas as pd
#from const import *
#from help_functions.run_R_code import run_r_script
from help_functions.help_simulation_pnl import simulate_pnl

#df_log_returns = create_log_returns("ETF_List.xlsx")

# Run the R script to perform Copula-GARCH
# Change 'script.R' with path to R code
#df_sim_returns = run_r_script('script.R', {'df_log_returns': df_log_returns})

#df_all_pnl = pd.DataFrame()
#df_pnl = simulate_pnl(df_sim_returns)
#df_all_pnl = pd.concat([df_all_pnl, df_pnl])

# Copula model
#
# df_sim_returns = pd.read_csv(File with simulated returns from R)
#
# df_all_violations = pd.DataFrame()
# for thold in [-100, -75, -50, -35, -25, -15, -10, -5]:
#   df_threshold_violations = calc_threshold_violation(df_sim_returns, threshold = thold)
#   df_all_violations = pd.concat[df_all_violations, df_threshold_violations]
