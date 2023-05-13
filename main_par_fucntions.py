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

df_hist_log_returns = create_log_returns("data/ETF_List.xlsx")
SIM_NUMBER = 1000
THRESHOLDS = [0.01, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
THRESHOLDS_LOG = np.log([x + 1 for x in THRESHOLDS])
CVAR_LEVEL = [0.1, 0.5, 1, 5, 10]
NUM_BOOTSTRAPPING = 100


# Define the function to be executed in parallel
def run_simulation(number_simulation, data):
    arr_sim_log_returns = call_r(
        script_content=script_content,
        inputs={"data": data},
        output_var="simulated_returns",
    )
    df_sim_log_returns = pd.DataFrame(
        {col: np.ravel(arr) for col, arr in arr_sim_log_returns.items()}
    )
    print("Done with simulation", number_simulation + 1)
    return df_sim_log_returns


# Open the script file and read its content
with open("ARMA_GARCH_FINAL_12052023.R", "r") as f:
    script_content = f.read()


def calculate_threshold_results(sim_results, thresholds):
    """
    Calculate threshold results based on simulation results and thresholds.

    Args:
        sim_results (list): List of simulation results (dataframes).
        thresholds (list): List of threshold values.

    Returns:
        threshold_results_dict (dict): Dictionary containing threshold results for each threshold.
    """
    threshold_results_dict = {}

    for i, threshold in enumerate(thresholds):
        threshold_results = []

        for sim_result in sim_results:
            df_defaults = calc_threshold_violation(sim_result, threshold=threshold)
            threshold_results.append(df_defaults)

        df_threshold = pd.concat(threshold_results)

        var_name = f"df_default_threshold_{abs(int(threshold * 100))}"
        threshold_results_dict[var_name] = df_threshold

    return threshold_results_dict


def calculate_default_probabilities(threshold_results_dict, sim_number):
    """
    Calculate default probabilities for each threshold based on threshold results.

    Args:
        threshold_results_dict (dict): Dictionary containing threshold results for each threshold.
        sim_number (int): Number of simulations.

    Returns:
        df_default_probabilities (pandas.DataFrame): DataFrame containing default probabilities for each threshold.
    """
    df_default_probabilities = pd.DataFrame(columns=lst_ticker_column_names)

    for var_name, df_threshold in threshold_results_dict.items():
        default_prob = df_threshold.sum() / sim_number

        default_prob_name = var_name.replace("df_default", "default")
        exec(f"{default_prob_name} = default_prob")

        df_default_probabilities.loc[default_prob_name] = default_prob

    return df_default_probabilities


def calculate_default_probabilities_wrapper(sim_results, thresholds, sim_number):
    """
    Wrapper function to calculate default probabilities based on simulation results and thresholds.

    Args:
        sim_results (list): List of simulation results (dataframes).
        thresholds (list): List of threshold values.
        sim_number (int): Number of simulations.

    Returns:
        df_default_probabilities (pandas.DataFrame): DataFrame containing default probabilities for each threshold.
    """
    threshold_results_dict = calculate_threshold_results(sim_results, thresholds)
    df_default_probabilities = calculate_default_probabilities(
        threshold_results_dict, sim_number
    )

    return df_default_probabilities


if __name__ == "__main__":
    df_all_CVaR = pd.DataFrame()
    df_all_default_prob_port = pd.DataFrame()
    for i in range(NUM_BOOTSTRAPPING):
        # Create a bootstrapped sample of the log returns.
        bootstrapped_returns = bootstrap_returns(df=df_hist_log_returns)

        # Determine the number of processes to use
        num_processes = multiprocessing.cpu_count()

        # Create a Pool object
        pool = multiprocessing.Pool(processes=num_processes)

        # Perform the simulation once
        sim_results = pool.starmap(
            run_simulation, [(x, bootstrapped_returns) for x in range(SIM_NUMBER)]
        )

        # Close the Pool
        pool.close()
        pool.join()

        lst_ticker_column_names = sim_results[0].columns.tolist()

        print("Done with simulating log returns and threshold verification.")

        df_default_probabilities = calculate_default_probabilities_wrapper(
            sim_results, THRESHOLDS_LOG, SIM_NUMBER
        )

        # Convert log returns to standard returns and store the converted log returns
        # converted_returns_dict = {}

        df_all_weighted_avg_log_returns = pd.DataFrame()
        df_all_weighted_avg_standard_returns = pd.DataFrame()
        for i, sim_result in enumerate(sim_results):
            # Convert log returns to standard returns
            df_standard_returns = convert_returns(sim_result, bool_to_log=False)

            # Calculate weighted average across each row, mean because equally weighted!
            df_weighted_avg_standard_returns = calculate_portfolio_return(
                df_standard_returns
            )

            # Store the weighted average in a variable
            # var_name = f"df_standard_port_returns_{i + 1}"
            # globals()[var_name] = df_weighted_avg_standard_returns

            # Add the weighted average to the dictionary
            # converted_returns_dict[var_name] = df_weighted_avg_standard_returns

            # Convert standard returns back to log returns
            df_weighted_avg_log_returns = convert_returns(
                df_weighted_avg_standard_returns
            )

            # aggregate all simulated portfolio log returns into one df
            sim_name = f"Simulation {i + 1}"
            df_all_weighted_avg_log_returns = pd.concat(
                [df_all_weighted_avg_log_returns, df_weighted_avg_log_returns], axis=1
            )
            df_all_weighted_avg_log_returns.rename(columns={0: sim_name}, inplace=True)

            # aggregate all simulated portfolio standard returns into one df
            df_all_weighted_avg_standard_returns = pd.concat(
                [
                    df_all_weighted_avg_standard_returns,
                    df_weighted_avg_standard_returns,
                ],
                axis=1,
            )
            df_all_weighted_avg_standard_returns.rename(
                columns={0: sim_name}, inplace=True
            )

        # calculate the default probability of the overall portfolio
        df_weighted_avg_log_threshold_viol = pd.DataFrame()
        df_weighted_avg_log_default_prob = pd.DataFrame(columns=["default_prob_portf"])
        for threshold in THRESHOLDS_LOG:
            df_defaults_portf = calc_threshold_violation(
                df_all_weighted_avg_log_returns, threshold
            )
            df_weighted_avg_log_threshold_viol = pd.concat(
                [df_weighted_avg_log_threshold_viol, df_defaults_portf]
            )

        df_weighted_avg_log_default_prob["default_prob_portf"] = (
            df_weighted_avg_log_threshold_viol.sum(axis=1) / SIM_NUMBER
        )
        df_weighted_avg_log_default_prob.index = [
            f"Threshold_{threshold * 100}" for threshold in THRESHOLDS
        ]

        # calculate the CVaR
        CVaR_estimations = []
        for level in CVAR_LEVEL:
            CVaR = get_CVaR(df_all_weighted_avg_standard_returns, level)
            CVaR_estimations.append(CVaR)

        CVaR = pd.DataFrame(
            {
                "CVaR_{}%".format(100 - level): [cvar]
                for level, cvar in zip(CVAR_LEVEL, CVaR_estimations)
            }
        )

        df_all_CVaR = pd.concat([df_all_CVaR, CVaR])
        df_all_default_prob_port = pd.concat(
            [df_all_default_prob_port, df_weighted_avg_log_default_prob], axis=1
        )

        df_all_CVaR.to_excel("df_all_CVaR.xlsx")
        df_all_default_prob_port.to_excel("df_all_default_prob_port.xlsx")
