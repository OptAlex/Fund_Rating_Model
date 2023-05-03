import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm, t


def get_fund_dict(df: pd.DataFrame) -> dict:
    """
    Create a dictionary of all funds with price & returns.
    """
    fund_dict = {}
    for col in df.columns:
        if col.startswith("price_"):
            fund_name = col.split("_")[1]
            prices = df[col].values
            returns = df[f"return_{fund_name}"].values
            fund_dict[fund_name] = (prices, returns)
    return fund_dict


def estimate_default_prob(
    df: pd.DataFrame,
    threshold: float,
    num_samples: int,
    window_size: int,
    step: int,
    score: str,
) -> pd.DataFrame:
    """Estimate the probability to default for each fund.

    To estimate the probability to default for each fund a rowling window approach is used. For bootstrapping
    the number of samples can be specified as needed. The probability to default is calculated by using the z- or t- score
    (depends on the assumption).
    """

    default_prob_dict = {}
    date = np.array(df["date"].values)
    fund_dict = get_fund_dict(df)

    for fund_name, (prices, returns) in fund_dict.items():
        default_prob = np.zeros(len(df))

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
            sigma2 = results.conditional_volatility**2

            bootstrap_returns = np.zeros((num_samples, len(returns)))
            for j in range(num_samples):
                bootstrap_start_idx = np.random.randint(
                    start_idx, end_idx - window_size + 1
                )
                bootstrap_end_idx = bootstrap_start_idx + window_size
                bootstrap_returns[j, :] = fund_dict[fund_name][1][
                    bootstrap_start_idx:bootstrap_end_idx
                ]

            bootstrap_volatility = np.zeros(num_samples)
            for j in range(num_samples):
                bootstrap_model = arch_model(
                    bootstrap_returns[j, :], p=1, q=1, rescale=False
                )
                bootstrap_results = bootstrap_model.fit(disp="off")
                bootstrap_volatility[j] = (
                    bootstrap_results.forecast(horizon=1, reindex=False)
                    .variance.iloc[-1]
                    .item()
                )

            for j in range(start_idx, end_idx):
                if j == start_idx:
                    start_value = fund_dict[fund_name][0][j]
                    current_value = start_value
                    current_volatility = sigma2[j]
                    current_date = df["date"][j]
                else:
                    current_value = fund_dict[fund_name][0][j]
                    current_date = df["date"][j]

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

            default_prob[start_idx:end_idx] /= num_samples

        default_prob_dict[fund_name] = default_prob

    default_prob_df = pd.DataFrame(default_prob_dict)
    default_prob_df["date"] = date
    default_prob_df = default_prob_df.set_index("date")
    return default_prob_df
