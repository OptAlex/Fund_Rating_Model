from help_functions.help_data_transformation import (
    create_log_returns,
    convert_returns,
    calculate_portfolio_return,
)
from help_functions.help_functions_default_prob import calc_threshold_violation
from help_functions.help_bootstrap import bootstrap_returns
from help_functions.help_CVaR import get_CVaR
import pandas as pd
from help_functions.run_R_code import call_r
import numpy as np
import multiprocessing
import datetime
import json
import os

df_hist_log_returns = create_log_returns("data/ETF_List.xlsx")
THRESHOLDS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085,
              0.09, 0.095, 0.10, 0.125, 0.15, 0.20, 0.3, 0.5]
THRESHOLDS_LOG = np.log([x + 1 for x in THRESHOLDS])
CVAR_LEVEL = [0.1, 0.5, 1, 2.5, 5, 10]
NUM_BOOTSTRAPPING = 2
BOOL_TO_EXCEL = True


# Define the function to be executed in parallel

def run_simulation(data):
    arr_sim_log_returns = call_r(
        script_content=script_content,
        inputs={"data": data},
        output_var="total_simulations",
    )
    df_sim_log_returns = pd.DataFrame(
        {col: np.ravel(arr) for col, arr in arr_sim_log_returns.items()}
    )
    print("Done with simulation")
    return df_sim_log_returns


# Open the script file and read its content
with open("ARMA_GARCH_FINAL_12052023.R", "r") as f:
    script_content = f.read()


def calc_default_probs(dfs, bool_indiv=False):
    # Default prob of individual funds
    df_all_defaults = pd.DataFrame()
    for threshold in THRESHOLDS:
        for i, df in enumerate(dfs):
            sim_results = df

            df_default = calc_threshold_violation(df=sim_results, threshold=threshold)
            str_threshold = f"Threshold_{round(threshold * 100, 2)}%"
            if threshold < 0.1:
                str_threshold = f"Threshold_0{round(threshold * 100, 2)}%"
            df_default.index = [str_threshold]
            df_all_defaults = pd.concat([df_all_defaults, df_default])

    if bool_indiv:
        df_all_def_prob = df_all_defaults.groupby(df_all_defaults.index).mean()  # default prob
        df_all_def_prob_sorted = df_all_def_prob.sort_index(axis=0)

    else:
        df_all_def_prob = df_all_defaults.sum(axis=1) / df_all_defaults.shape[1]
        df_all_def_prob_sorted = df_all_def_prob.sort_index(axis=0)

    return df_all_def_prob_sorted


if __name__ == "__main__":
    df_all_CVaR = pd.DataFrame()
    df_all_default_prob_port = pd.DataFrame()
    times = []
    for i in range(NUM_BOOTSTRAPPING):
        current_time = datetime.datetime.now()
        print(f"Current time bootstrap {i}:", current_time)
        times.append(current_time)

        # Create a bootstrapped sample of the log returns.
        bootstrapped_returns = bootstrap_returns(df=df_hist_log_returns)
        _ = run_simulation(data=bootstrapped_returns)

        with open('total_simulations.json', 'r') as file:
            json_data = json.load(file)

        # Extract each individual data frame
        df_simulations = []
        for data_frame in json_data:
            df = pd.DataFrame(data_frame)
            df_simulations.append(df)

        df_indiv_fonds_default_prob = calc_default_probs(dfs=df_simulations, bool_indiv=True)

        # calculate portfolio defaults
        df_all_weighted_avg_log_portf_returns = pd.DataFrame()
        df_all_weighted_avg_std_portf_returns = pd.DataFrame()
        for i, df in enumerate(df_simulations):
            df_standard_returns = convert_returns(df, bool_to_log=False)
            df_weighted_avg_standard_portf_returns = calculate_portfolio_return(df_standard_returns)
            df_all_weighted_avg_log_portf_returns[f"Sim_{i}"] = convert_returns(df_weighted_avg_standard_portf_returns)
            df_all_weighted_avg_std_portf_returns[f"Sim_{i}"] = df_weighted_avg_standard_portf_returns

        df_portf_default_prob = calc_default_probs([df_all_weighted_avg_log_portf_returns])

        # calculate the CVaR
        CVaR_estimations = []
        for level in CVAR_LEVEL:
            CVaR = get_CVaR(df_all_weighted_avg_std_portf_returns, level)
            CVaR_estimations.append(CVaR)

        CVaR = pd.DataFrame(
            {
                "CVaR_{}%".format(100 - level): [cvar]
                for level, cvar in zip(CVAR_LEVEL, CVaR_estimations)
            }
        )
        df_all_CVaR = pd.DataFrame()
        df_all_CVaR = pd.concat([df_all_CVaR, CVaR])

        if BOOL_TO_EXCEL:
            df_indiv_fonds_default_prob.to_excel("df_indiv_fonds_default_prob_one_copula.xlsx")
            df_indiv_fonds_default_prob.to_excel("df_portf_default_prob_one_copula.xlsx")
            df_all_CVaR.to_excel("df_all_CVaR_one_copula_without_stopp_loss.xlsx")

    print(times)
