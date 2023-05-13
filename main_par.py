from help_functions.help_data_transformation import create_log_returns, convert_returns, calculate_portfolio_return
from help_functions.help_functions_default_prob import calc_threshold_violation
from help_functions.help_bootstrap import bootstrap_returns
from help_functions.help_CVaR import get_CVaR
import pandas as pd
from help_functions.run_R_code import call_r
import numpy as np
import multiprocessing

df_hist_log_returns = create_log_returns("data/ETF_List.xlsx")
SIM_NUMBER = 5
THRESHOLDS = [-0.01, -0.05, -0.10, -0.15, -0.20, -0.30, -0.50]
CVaR_level = [0.1, 0.5, 1, 5, 10]
bool_to_excel = False

# Create a bootstrapped sample of the log returns.
bootstrapped_returns = bootstrap_returns(df=df_hist_log_returns, bootstrap_samples=1, bootstrap_days=252)

#ToDO Alex, this is the final CVaR calc. You can put the code to the right place (bootstrap_returns was our dummy).
# currently we on get the CVaRs. Do we also need a distribution or plot?
# Should we use the standard returns for the CVaR? Or should we go consistent with log?
CVaR_estimations = []
for level in CVaR_level:
    CVaR = get_CVaR(bootstrapped_returns, level)
    CVaR_estimations.append(CVaR)

CVaR = pd.DataFrame({'CVaR_{}%'.format(100-level): [cvar] for level, cvar in zip(CVaR_level, CVaR_estimations)})

# Define the function to be executed in parallel
def run_simulation(number_simulation):
    arr_sim_log_returns = call_r(
        script_content=script_content,
        inputs={"df_log_returns": df_hist_log_returns},
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

if __name__ == "__main__":
    # Determine the number of processes to use
    num_processes = multiprocessing.cpu_count()

    # Create a Pool object
    pool = multiprocessing.Pool(processes=num_processes)

    # Perform the simulation once
    sim_results = pool.map(run_simulation, range(SIM_NUMBER))

    # Close the Pool
    pool.close()
    pool.join()

    print("Done with simulating log returns and threshold verification.")

    # Create a dictionary to store the threshold results
    threshold_results_dict = {}

    # Write the results to Excel
    with pd.ExcelWriter("threshold_results.xlsx") as writer:
        # Write each threshold result to a separate sheet
        for i, threshold in enumerate(THRESHOLDS):
            threshold_results = []

            for sim_result in sim_results:
                df_defaults = calc_threshold_violation(sim_result, threshold=threshold)
                threshold_results.append(df_defaults)

            sheet_name = f"Threshold {threshold}"
            df_threshold = pd.concat(threshold_results)
            df_threshold.to_excel(writer, sheet_name=sheet_name, index=False)

            # Store the DataFrame with the corresponding threshold in a variable
            var_name = f"df_default_threshold_{abs(int(threshold * 100))}"
            threshold_results_dict[var_name] = df_threshold

    print("Threshold results written to Excel.")

    # Calculate default probabilities for each threshold
    df_default_probabilities = pd.DataFrame(columns=df_defaults.columns)

    for var_name, df_threshold in threshold_results_dict.items():
        # Sum the data by column and divide each sum by the number of simulations
        default_prob = df_threshold.sum() / SIM_NUMBER

        # Save the result as default_prob_x
        default_prob_name = var_name.replace("df_default", "default")
        exec(f"{default_prob_name} = default_prob")

        # Add default_prob to the DataFrame
        df_default_probabilities.loc[default_prob_name] = default_prob

    df_default_probabilities.to_excel("default_probabilities.xlsx", index=True)

    # Convert log returns to standard returns and store the converted log returns
    converted_returns_dict = {}

    # Create a writer for the Excel file
    converted_returns_writer = pd.ExcelWriter("converted_log_returns.xlsx")

    for i, sim_result in enumerate(sim_results):
        # Convert log returns to standard returns
        df_standard_returns = convert_returns(sim_result, bool_to_log=False)

        # Calculate weighted average across each row, mean because equally weighted!
        df_weighted_avg_standard_returns = calculate_portfolio_return(df_standard_returns)

        # Store the weighted average in a variable
        var_name = f"df_standard_port_returns_{i + 1}"
        globals()[var_name] = df_weighted_avg_standard_returns

        # Add the weighted average to the dictionary
        converted_returns_dict[var_name] = df_weighted_avg_standard_returns

        # Convert standard returns back to log returns
        df_weighted_avg_log_returns = convert_returns(df_weighted_avg_standard_returns)

        # Save the log returns to the Excel writer with a separate sheet for each simulation
        sheet_name = f"Simulation {i + 1}"
        df_weighted_avg_log_returns.to_excel(converted_returns_writer, sheet_name=sheet_name, index=False)

    # Save the Excel file with all the converted log returns
    converted_returns_writer.save()

    # ToDo: df_weighted_avg_log_returns are the log returns for each simulation of the overall portfolio. Plug them
    #  into the threshold violation to calc the portfolio default prob

    # Save all converted log returns to a single Excel file with different sheets
    with pd.ExcelWriter("converted_log_returns.xlsx") as writer:
        for var_name, df_log_returns in converted_returns_dict.items():
            sheet_name = var_name.replace("df_standard_port_returns_", "Simulation ")
            df_log_returns.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save sim_results to an Excel file with one sheet per simulation
    with pd.ExcelWriter("sim_results.xlsx") as writer:
        for i, sim_result in enumerate(sim_results):
            sheet_name = f"Simulation {i + 1}"
            sim_result.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Results written to Excel files.")


##

