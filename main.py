import io
from help_functions.help_data_transformation import (
    create_log_returns,
    convert_returns,
    calculate_portfolio_return,
)
from help_functions.help_functions_default_prob import calc_default_probs
from help_functions.help_bootstrap import bootstrap_returns
from help_functions.help_CVaR import get_CVaR
import pandas as pd
from help_functions.help_R_code import call_r
import datetime
import json
import multiprocessing
from const import *

df_hist_log_returns = create_log_returns("data/ETF_List.xlsx")
times = []


def run_simulation(data):
    arr_sim_log_returns = call_r(
        script_content=script_content, inputs={"data": data}, output_var="json_data"
    )
    print("Done with simulation")

    return arr_sim_log_returns


# Open the script file and read its content
with open("ARMA_GARCH.R", "r") as f:
    script_content = f.read()


# Define the function to execute each iteration
def process_iteration(i):
    """
    Includes the whole process to simulate PD and CVaR
    :param i: itteration number
    :return: CVaR, CVaR_estimations_stop_loss, df_portf_default_prob, df_indiv_fonds_default_prob,
    """
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

    # both, standard and log returns are needed due to their individual properties
    df_weighted_avg_log_portf_returns = []
    df_weighted_avg_std_portf_returns = []

    for df in df_simulations:
        df_standard_returns = convert_returns(df, bool_to_log=False)
        df_standard_portf_returns = calculate_portfolio_return(df_standard_returns)
        df_weighted_avg_log_portf_returns.append(
            convert_returns(df_standard_portf_returns)
        )
        df_weighted_avg_std_portf_returns.append(df_standard_portf_returns)

    df_all_weighted_avg_log_portf_returns = pd.concat(
        df_weighted_avg_log_portf_returns,
        axis=1,
        keys=[f"Sim_{i}" for i in range(len(df_simulations))],
    )
    df_all_weighted_avg_std_portf_returns = pd.concat(
        df_weighted_avg_std_portf_returns,
        axis=1,
        keys=[f"Sim_{i}" for i in range(len(df_simulations))],
    )

    # calculate portfolio default probabilities
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
    for i in range(1):
        # We use try and except because when bootstrapping the data we sometimes end up with errors regarding
        # singularity and the Hessian matrix used for optimisation. Even though further research on this would be
        # possible and different optimisation methods could be applied based on the data used, we think this is
        # beyond the scope of our project. Because this error results in complete loss of all other
        # simulations, we decided to implement a try and except for 100 loops, 10 bootstrapping (because of a 10
        # core machine) for each loop. This allowed us to gather 580 bootstrapping samples (58 loops).
        try:
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
            df_all_CVaR_estimations_stop_loss = pd.concat(
                all_CVaR_estimations_stop_loss
            )
            df_all_default_prob_port = pd.concat(all_default_prob_port_results, axis=1)
            df_all_indiv_fonds_default_prob = pd.concat(
                all_indiv_fonds_default_prob_results, axis=1
            )

            if BOOL_TO_EXCEL:
                df_all_default_prob_port.to_excel(
                    f"df_portf_default_prob_no_boot_{i}.xlsx"
                )
                df_all_CVaR.to_excel(f"df_all_CVaR_without_stop_loss_no_boot_{i}.xlsx")
                df_all_CVaR_estimations_stop_loss.to_excel(
                    f"df_all_CVaR_stop_loss_{i}.xlsx"
                )

                # Individual default probs to excel. Only for the first 100 because of sheet limitations of Excel
                writer = pd.ExcelWriter(
                    f"df_indiv_fonds_default_prob_no_boot_{i}.xlsx", engine="xlsxwriter"
                )
                for k, df in enumerate(all_indiv_fonds_default_prob_results[:100]):
                    df.to_excel(writer, sheet_name=f"Simulation_{k + 1}", index=True)
                writer._save()

        except:
            pass
