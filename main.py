import io

from help_functions.help_confidence_interval import (
    get_confidence_interval,
    confidence_interval_no_distr,
)
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
import datetime
import json
import multiprocessing

df_hist_log_returns = create_log_returns("data/ETF_List.xlsx")
THRESHOLDS = [
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.035,
    0.04,
    0.045,
    0.05,
    0.055,
    0.06,
    0.065,
    0.07,
    0.075,
    0.08,
    0.085,
    0.09,
    0.095,
    0.10,
    0.125,
    0.15,
    0.20,
    0.3,
    0.5,
]
THRESHOLDS_LOG = np.log([x + 1 for x in THRESHOLDS])
CVAR_LEVEL = [0.1, 0.5, 1, 2.5, 5, 10]
CI_LEVELS = [0.999, 0.99, 0.975, 0.95, 0.90]
NUM_BOOTSTRAPPING = 100
STOP_LOSS = 0.075
BOOL_TO_EXCEL = True
times = []


def run_simulation(data):
    arr_sim_log_returns = call_r(
        script_content=script_content,
        inputs={"data": data},
        output_var="json_data",
    )
    # df_sim_log_returns = pd.DataFrame(
    #     {col: np.ravel(arr) for col, arr in arr_sim_log_returns.items()}
    # )
    print("Done with simulation")

    return arr_sim_log_returns


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
        df_all_def_prob = df_all_defaults.groupby(
            df_all_defaults.index
        ).mean()  # default prob
        df_all_def_prob_sorted = df_all_def_prob.sort_index(axis=0)

    else:
        df_all_def_prob = df_all_defaults.sum(axis=1) / df_all_defaults.shape[1]
        df_all_def_prob_sorted = df_all_def_prob.sort_index(axis=0)

    return df_all_def_prob_sorted


# Define the function to execute each iteration
def process_iteration(i):
    current_time = datetime.datetime.now()
    print(f"Current time bootstrap {i}:", current_time)

    # Create a bootstrapped sample of the log returns.
    bootstrapped_returns = bootstrap_returns(df=df_hist_log_returns)
    sim_json = run_simulation(data=bootstrapped_returns)

    # Convert sim_json to a file-like object
    sim_json_file = io.BytesIO(sim_json.tobytes())

    sim_log_returns = json.load(sim_json_file)

    # Extract each individual data frame
    df_simulations = []
    for data_frame in sim_log_returns:
        df = pd.DataFrame(data_frame)
        df_simulations.append(df)

    df_indiv_fonds_default_prob = calc_default_probs(
        dfs=df_simulations, bool_indiv=True
    )

    df_simulations_converted = []
    df_simulations_weighted_avg = []

    for df in df_simulations:
        df_standard_returns = convert_returns(df, bool_to_log=False)
        df_weighted_avg_standard_portf_returns = calculate_portfolio_return(
            df_standard_returns
        )
        df_simulations_converted.append(
            convert_returns(df_weighted_avg_standard_portf_returns)
        )
        df_simulations_weighted_avg.append(df_weighted_avg_standard_portf_returns)

    df_all_weighted_avg_log_portf_returns = pd.concat(
        df_simulations_converted,
        axis=1,
        keys=[f"Sim_{i}" for i in range(len(df_simulations))],
    )
    df_all_weighted_avg_std_portf_returns = pd.concat(
        df_simulations_weighted_avg,
        axis=1,
        keys=[f"Sim_{i}" for i in range(len(df_simulations))],
    )

    df_portf_default_prob = calc_default_probs([df_all_weighted_avg_log_portf_returns])

    # calculate the CVaR
    CVaR_estimations = []
    CVaR_estimations_stop_loss = []
    for level in CVAR_LEVEL:
        CVaR = get_CVaR(df_all_weighted_avg_std_portf_returns, level)
        CVaR_stop_loss = get_CVaR(
            df_all_weighted_avg_std_portf_returns,
            level,
            stop_loss=True,
            threshold=STOP_LOSS,
        )
        CVaR_estimations.append(CVaR)
        CVaR_estimations_stop_loss.append(CVaR_stop_loss)

    CVaR = pd.DataFrame(
        {
            "CVaR_{}%".format(100 - level): [cvar]
            for level, cvar in zip(CVAR_LEVEL, CVaR_estimations)
        }
    )

    CVaR_estimations_stop_loss = pd.DataFrame(
        {
            "CVaR_SL_{}%".format(100 - level): [cvar]
            for level, cvar in zip(CVAR_LEVEL, CVaR_estimations_stop_loss)
        }
    )

    return (
        CVaR,
        CVaR_estimations_stop_loss,
        df_portf_default_prob,
        df_indiv_fonds_default_prob,
    )


if __name__ == "__main__":
    # Parallelize the for loop
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_iteration, range(NUM_BOOTSTRAPPING))

    # Extract the results from the pool
    (
        all_CVaR_results,
        all_CVaR_estimations_stop_loss,
        all_default_prob_port_results,
        all_indiv_fonds_default_prob_results,
    ) = zip(*results)

    # Concatenate the results
    df_all_CVaR = pd.concat(all_CVaR_results)
    df_all_CVaR_estimations_stop_loss = pd.concat(all_CVaR_estimations_stop_loss)
    df_all_default_prob_port = pd.concat(all_default_prob_port_results, axis=1)
    df_all_indiv_fonds_default_prob = pd.concat(
        all_indiv_fonds_default_prob_results, axis=1
    )

    df_CVar_CI = confidence_interval_no_distr(data=df_all_CVaR, alpha=CI_LEVELS)
    df_CVar_stop_loss_CI = confidence_interval_no_distr(
        data=df_all_CVaR, alpha=CI_LEVELS
    )
    df_portf_default_prob_CI = confidence_interval_no_distr(
        data=df_all_default_prob_port.transpose(), alpha=CI_LEVELS
    )
    df_indiv_funds_default_prob_CI = confidence_interval_no_distr(
        data=df_all_indiv_fonds_default_prob.transpose(), alpha=CI_LEVELS
    )

    if BOOL_TO_EXCEL:
        df_all_indiv_fonds_default_prob.to_excel(
            "df_indiv_fonds_default_prob_one_copula.xlsx"
        )
        df_all_default_prob_port.to_excel("df_portf_default_prob_one_copula.xlsx")
        df_all_CVaR.to_excel("df_all_CVaR_one_copula_without_stopp_loss.xlsx")

        df_CVar_CI.to_excel("CI_all_CVaR_one_copula_without_stop_loss.xlsx")
        df_CVar_stop_loss_CI.to_excel("CI_all_CVaR_one_copula_stop_loss.xlsx")
        df_portf_default_prob_CI.to_excel("CI_df_portf_default_prob_CI.xlsx")
        df_indiv_funds_default_prob_CI.to_excel(
            "CI_df_indiv_funds_default_prob_CI.xlsx"
        )
