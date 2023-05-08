import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from recombinator.block_bootstrap import circular_block_bootstrap
from recombinator.optimal_block_length import optimal_block_length
from help_functions.help_functions_default_prob import get_fund_dict, simulate_returns, run_garch_model, get_ci_from_df


def estimate_default_prob(
        df: pd.DataFrame,
        threshold: float,
        bootstrap_samples: int,
        simulation_samples: int,
        prediction_days: int,
) -> pd.DataFrame:
    """
    Perform a Monte Carlo Simulation and estimate the probability to default for each fund by counting the violations.

    The future default probability is estimated on future returns from a Monte Carlo Simulation. For bootstrapping
    the number of samples can be specified as needed. The probability to default is calculated by counting the violations
    of the specified threshold in the cumulative returns.
    :param df: df with the historical prices and returns
    :param threshold: threshold which defines the default of a fund
    :param bootstrap_samples: number of samples to bootstrap the data
    :param simulation_samples: number of samples to simulate the data
    :param prediction_days: defines the number of days to estimate the default probabilities
    :return: df with the predicted default probabilities
    """
    fund_dict = get_fund_dict(df)
    default_prob_dict = {fund_name: [] for fund_name in fund_dict.keys()}

    for fund_name, (prices, returns) in fund_dict.items():
        # Calculate optimal block length
        b_star = optimal_block_length(returns)
        b_star_cb = math.ceil(b_star[0].b_star_cb)

        # Perform bootstrapping of the data
        bootstrap_returns = circular_block_bootstrap(
            returns,
            block_length=b_star_cb,
            replications=bootstrap_samples,
            sub_sample_length=prediction_days,
            replace=True,
        )

        # Calculate the mean of the bootstrapped returns
        #bootstrap_returns = np.mean(bootstrap_returns, axis=0)
        bootstrap_returns = bootstrap_returns * 100

        for bootstrap_return in bootstrap_returns:
            # Fit the GARCH model to the returns
            res = run_garch_model(bootstrap_return)
            garch_parameters = res.params

            # Initialize array to store the simulated returns for each day
            simulated_returns_array = np.zeros((simulation_samples, prediction_days))

            for i in range(simulation_samples):
                # Simulate future returns using the GARCH model
                simulated_returns = simulate_returns(
                    garch_parameters, num_days_to_simulate=prediction_days
                )
                # Store the simulated returns for each day
                simulated_returns_array[i] = simulated_returns / 100

            # Calculate the cumulative returns for each day
            cumulative_returns_array = np.cumsum(simulated_returns_array, axis=1)

            # Count the number of times the threshold (negativ, therefore 1 + threshold) is violated in the cumulative returns
            num_violations = np.sum(np.any(cumulative_returns_array <= threshold, axis=1))

            # Calculate the default probability for the fund
            default_prob = num_violations / simulation_samples

            # Append the default probability for the fund to the list in the dictionary
            default_prob_dict[fund_name].append(default_prob)

    default_prob_columns = {f'{fund_name}_{prediction_days}_days_default_prob': prob_list for fund_name, prob_list in
                            default_prob_dict.items()}
    # Create a pandas dataframe to store the default probabilities
    default_prob_df = pd.DataFrame(default_prob_columns)

    # Return the default probabilities
    return default_prob_df

def plot_default_prob(df: pd.DataFrame, confidence_level: float):
    """
    Plot a histogram of the default probabilities of the portfolio and add the distribution as well as the confidence level.
    :param df: DataFrame with default probabilities for the portfolio.
    :param confidence_level: Desired confidence level.
    :return: Histogram plot.
    """
    ci_dict = get_ci_from_df(df, confidence_level)
    for col in df.columns:
        plt.figure()
        sns.displot(df[col], kde=True)#, shade=True, fill=True)
        plt.title(f"Distribution of {col} Default Probabilities")
        plt.xlabel("Default Probability")
        plt.ylabel("Density")
        # Calculate and plot the confidence interval
        lower, upper = ci_dict['confidence_interval'][col]
        plt.axvline(lower, color='red', linestyle='--')
        plt.axvline(upper, color='red', linestyle='--')
        plt.show()
