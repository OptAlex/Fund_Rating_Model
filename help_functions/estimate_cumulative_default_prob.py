import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm, t
from datetime import timedelta


############################################################################################
# Help functions for the estimation of the default probabilities
############################################################################################
def get_fund_dict(df: pd.DataFrame) -> dict:
    """
    Create a dictionary of all funds with price & returns.
    :param df: df with the historical prices and returns
    :return: dictionary with the fund name and the historical prices and returns of the fund
    """
    fund_dict = {}
    for col in df.columns:
        if col.startswith("price_"):
            fund_name = col.split("_")[1]
            prices = df[col].values
            returns = df[f"return_{fund_name}"].values
            fund_dict[fund_name] = (prices, returns)
    return fund_dict


def simulate_returns(garch_parameters, num_days_to_simulate, p=1, q=1):
    """
    Function to simulate the future returns
    :param q:
    :param p:
    :param garch_parameters: alpha, beta, sigma
    :param num_days_to_simulate: number of days on which the simulation is performed
    :return: simulated returns for the number of prediction days
    """
    sim_mod = arch_model(None, p=p, q=q, rescale="False")
    sim_data = sim_mod.simulate(garch_parameters, num_days_to_simulate)
    return sim_data["data"]


def run_garch_model(returns):
    """
    Simple function to run a GARCH model
    :param returns: returns on which to run the model
    :return: the residuals of the model
    """
    # using GARCH(1,1)
    am = arch_model(returns, p=1, q=1, rescale="False")
    res = am.fit(disp="off")
    return res


############################################################################################
# Backtest of the default probabilities
############################################################################################
def estimate_historical_stat_default_prob(
        df: pd.DataFrame,
        threshold: float,
        window_size: int,
        step: int,
        score: str,
) -> pd.DataFrame:
    """
    Estimate the probability to default for each fund by using a statistical approach.

    To estimate the probability to default for each fund a rowling window approach is used. The probability to default is calculated by using the z- or t- score
    (depends on the assumption).

    :param df: df with the historical prices and returns
    :param threshold: threshold which defines the default of a fund
    :param window_size: defines the days of the rolling window
    :param step: steps wo roll on in the rolling window
    :param score: assumed distribution of the returns
    :return: df with the historical default probabilities
    """
    default_prob_dict = {}
    date = np.array(df["date"].values)
    fund_dict = get_fund_dict(df)

    for fund_name, (prices, returns) in fund_dict.items():
        default_prob = np.zeros(len(df))

        # multiply returns by 100 as this is prefered by the GARCH package
        returns *= 100

        for i in range(0, len(df) - window_size, step):
            start_idx = i
            end_idx = i + window_size
            if end_idx > len(df):
                break

            returns = fund_dict[fund_name][1][start_idx:end_idx]

            model = arch_model(returns, p=1, q=1, rescale=False)
            results = model.fit(disp="off")
            alpha = results.params["alpha[1]"]
            beta = results.params["beta[1]"]
            sigma2 = results.conditional_volatility ** 2

            for j in range(start_idx, end_idx):
                if j == start_idx:
                    start_value = fund_dict[fund_name][0][j]
                    current_value = start_value
                    current_date = df["date"][j]
                else:
                    current_value = fund_dict[fund_name][0][j]
                    current_date = df["date"][j]

                # get the corresponding sigma2 value for j by adjusting for the start index
                current_volatility = sigma2[j - start_idx]

                default_value = start_value * (1 - threshold)
                if score == "z_score":
                    z_score = (
                                      np.log(default_value / current_value)
                                      + (alpha + beta * current_volatility) / 2
                              ) / np.sqrt(current_volatility)
                    default_prob[j] += norm.sf(z_score)
                else:
                    t_score = (
                                      np.log(default_value / current_value)
                                      + (alpha + beta * current_volatility) / 2
                              ) / np.sqrt(current_volatility / window_size)
                    default_prob[j] += 1 - t.cdf(t_score, window_size - 1)
                date[j] = current_date

        # Divide the default prob by 100 as we multiplied the returns earlier
        default_prob_dict[f"default_prob_{fund_name}"] = default_prob / 100

    default_prob_df = pd.DataFrame(default_prob_dict)
    default_prob_df["date"] = date
    default_prob_df = default_prob_df.set_index("date")
    # ToDo add a confidence intervall
    return default_prob_df


def estimate_historical_count_default_prob(
        df: pd.DataFrame,
        threshold: float,
        window_size: int,
        step: int,
) -> pd.DataFrame:
    """
    Estimate the probability to default for each fund by using a statistical approach.

    To estimate the probability to default for each fund a rowling window approach is used. The probability to default is calculated by using the z- or t- score
    (depends on the assumption).

    :param df: df with the historical prices and returns
    :param threshold: threshold which defines the default of a fund
    :param window_size: defines the days of the rolling window
    :param step: steps wo roll on in the rolling window
    :param score: assumed distribution of the returns
    :return: df with the historical default probabilities
    """
    default_prob_dict = {}
    date = np.array(df["date"].values)
    fund_dict = get_fund_dict(df)

    for fund_name, (prices, returns) in fund_dict.items():
        default_prob = np.zeros(len(df))

        # multiply returns by 100 as this is prefered by the GARCH package
        returns *= 100

        for i in range(0, len(df) - window_size, step):
            start_idx = i
            end_idx = i + window_size
            if end_idx > len(df):
                break

            returns = fund_dict[fund_name][1][start_idx:end_idx]

            for j in range(len(returns)):
                # Calculate the threshold violations
                if returns[j] / 100 <= -threshold:
                    default_prob[j] += 1
                    print(default_prob)

            # Divide the default prob by 100 as we multiplied the returns earlier
            default_prob_dict[f"default_prob_{fund_name}"] = default_prob / 100

    default_prob_df = pd.DataFrame(default_prob_dict)
    default_prob_df["date"] = date
    default_prob_df = default_prob_df.set_index("date")
    # ToDo add a confidence intervall
    return default_prob_df


############################################################################################
# Forecast of the default probabilities
############################################################################################
def estimate_mc_stat_default_prob(
        df: pd.DataFrame,
        threshold: float,
        num_samples: int,
        prediction_days: int,
        score: str,
) -> pd.DataFrame:
    """
    Perform a Monte Carlo Simulation and estimate the probability to default for each fund by using a statistical approach.

    The future default probability is estimated on future log returns from a Monte Carlo Simulation. For bootstrapping
    the number of samples can be specified as needed.
    The probability to default is calculated by using the z- or t- score (depends on the assumed distribution).

    :param df: df with the historical prices and returns
    :param threshold: threshold which defines the default of a fund
    :param num_samples: number of samples to bootstrap the data
    :param prediction_days: defines the number of days to estimate the default probabilities
    :param score: assumed distribution of the returns
    :return: df with the predicted default probabilities
    """
    default_prob_dict = {}
    fund_dict = get_fund_dict(df)

    last_day = pd.to_datetime(df["date"].iloc[-1])
    end_date = last_day + timedelta(days=prediction_days - 1)

    for fund_name, (prices, returns) in fund_dict.items():
        default_prob = np.zeros(prediction_days)

        # Multiply returns by 100 as this is preferred by the GARCH package
        returns *= 100

        # Fit the GARCH model to the returns
        res = run_garch_model(returns)
        garch_parameters = res.params

        for i in range(num_samples):
            # Simulate future returns using the GARCH model
            simulated_returns = simulate_returns(
                garch_parameters, num_days_to_simulate=prediction_days
            )

            # Calculate the conditional volatility for the simulated returns
            sim_mod = arch_model(simulated_returns, p=1, q=1, rescale="False")
            sim_res = sim_mod.fit(disp="off")
            alpha = sim_res.params["alpha[1]"]
            beta = sim_res.params["beta[1]"]
            sigma2 = sim_res.conditional_volatility ** 2

            # Calculate the default probability for each day in the future
            start_value = prices[-1]
            for j in range(prediction_days):
                current_value = start_value * (1 + simulated_returns[j] / 100)
                current_volatility = sigma2[j]
                default_value = start_value * (1 - threshold)

                if score == "z_score":
                    z_score = (
                                      np.log(default_value / current_value)
                                      + (alpha + beta * current_volatility) / 2
                              ) / np.sqrt(current_volatility)
                    default_prob[j] += norm.sf(z_score)
                else:
                    t_score = (
                                      np.log(default_value / current_value)
                                      + (alpha + beta * current_volatility) / 2
                              ) / np.sqrt(current_volatility / prediction_days)
                    default_prob[j] += 1 - t.cdf(t_score, prediction_days - 1)

                # Update the start value for the next day
                start_value = current_value

        # Divide the default prob by the number of samples
        default_prob /= num_samples

        # Add the default probabilities for the fund to the dictionary
        # Divide the default prob by 100 as we multiplied the returns earlier
        default_prob_dict[f"default_prob_{fund_name}"] = default_prob / 100

    # Create a date range for the specified days
    date_range = pd.date_range(start=last_day, end=end_date)

    # Create a pandas dataframe to store the default probabilities
    default_prob_df = pd.DataFrame(default_prob_dict, index=date_range)

    # Resample the dataframe to get the default probabilities for each day
    daily_default_prob_df = default_prob_df.resample("D").asfreq()

    # Forward fill missing values
    daily_default_prob_df.fillna(method="ffill", inplace=True)

    # Return the default probabilities for the next month
    return daily_default_prob_df


def estimate_mc_count_cumulative_return(
        df: pd.DataFrame,
        threshold: float,
        num_samples: int,
        prediction_days: int,
) -> pd.DataFrame:
    """
    Perform a Monte Carlo Simulation and estimate the probability to default for each fund by counting the violations.

    The future default probability is estimated on future returns from a Monte Carlo Simulation. For bootstrapping
    the number of samples can be specified as needed. The probability to default is calculated by counting the violations
    of the specified threshold in the cumulative returns.
    :param df: df with the historical prices and returns
    :param threshold: threshold which defines the default of a fund
    :param num_samples: number of samples to bootstrap the data
    :param prediction_days: defines the number of days to estimate the default probabilities
    :return: df with the predicted default probabilities
    """
    default_prob_dict = {}
    fund_dict = get_fund_dict(df)

    for fund_name, (prices, returns) in fund_dict.items():
        default_prob = np.zeros(prediction_days)

        sample_size = 252
        bootstrap_returns = []
        for i in range(num_samples):
            # Randomly select 252 returns from the data, with replacement
            subset_indices = np.random.choice(len(returns), size=sample_size, replace=True)
            subset = returns[subset_indices]
            bootstrap_returns.append(subset)

        # Calculate the mean of the bootstrapped returns
        bootstrap_returns = np.mean(bootstrap_returns, axis=0)
        bootstrap_returns = bootstrap_returns * 100

        # Fit the GARCH model to the returns
        res = run_garch_model(bootstrap_returns)
        garch_parameters = res.params

        # Initialize array to store the simulated returns for each day
        simulated_returns_array = np.zeros((num_samples, prediction_days))

        for i in range(num_samples):
            # Simulate future returns using the GARCH model
            simulated_returns = simulate_returns(
                garch_parameters, num_days_to_simulate=prediction_days
            )
            # Store the simulated returns for each day
            simulated_returns_array[i] = simulated_returns / 100

        # Calculate the cumulative returns for each day
        cumulative_returns_array = np.cumprod(1 + simulated_returns_array, axis=1)

        # Count the number of times the threshold (negativ, therefore 1 + threshold) is violated in the cumulative returns
        num_violations = np.sum(np.any(cumulative_returns_array <= 1 + threshold, axis=1))

        # Calculate the default probability for the fund
        default_prob = num_violations / num_samples

        # Add the default probability for the fund to the dictionary
        default_prob_dict[fund_name] = default_prob

    # Create a pandas dataframe to store the default probabilities
    default_prob_df = pd.DataFrame.from_dict(default_prob_dict, orient='index', columns=['one_month_default_prob'])

    # Return the default probabilities
    return default_prob_df


