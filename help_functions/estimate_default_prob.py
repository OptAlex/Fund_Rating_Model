import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm, t


def estimate_default_prob(df: pd.DataFrame, threshold: float, num_samples: int, window_size: int, step: int, score: str) -> pd.DataFrame:
    # Initialize default probability array
    default_prob = np.zeros(len(df))

    # Initialize date array
    date = np.array(df['date'].values)

    # Iterate through sliding windows
    for i in range(0, len(df) - window_size, step):
        # Define window start and end indices
        start_idx = i
        end_idx = i + window_size
        if end_idx > len(df):
            break  # Break the loop if end_idx exceeds the length of the DataFrame

        # Get log returns in window
        returns = df['return_agg'][start_idx:end_idx]

        # Fit GARCH model to log returns
        model = arch_model(returns, p=1, q=1, rescale=False)
        results = model.fit(disp='off')
        alpha = results.params['alpha[1]']
        beta = results.params['beta[1]']
        sigma2 = results.conditional_volatility ** 2

        # Bootstrap the data using sliding window
        bootstrap_returns = np.zeros((num_samples, len(returns)))
        for j in range(num_samples):
            # Define bootstrap window start and end indices
            bootstrap_start_idx = np.random.randint(start_idx, end_idx - window_size + 1)
            bootstrap_end_idx = bootstrap_start_idx + window_size

            # Get log returns in bootstrap window
            bootstrap_returns[j, :] = df['return_agg'][bootstrap_start_idx:bootstrap_end_idx]

        # Estimate volatility for each bootstrap sample
        bootstrap_volatility = np.zeros(num_samples)
        for j in range(num_samples):
            bootstrap_model = arch_model(bootstrap_returns[j, :], p=1, q=1, rescale=False)
            bootstrap_results = bootstrap_model.fit(disp='off')
            bootstrap_volatility[j] = bootstrap_results.forecast(horizon=1, reindex=False).variance.iloc[-1].item()

        # Estimate default probability using statistical method
        for j in range(start_idx, end_idx):
            if j == start_idx:
                start_value = df['price_agg'][j]
                current_value = start_value
                current_volatility = sigma2[j]
                current_date = df['date'][j]
            else:
                current_value = df['price_agg'][j]
                current_date = df['date'][j]

            default_value = start_value * (1 - threshold)
            if score == 'z_score':
                z_score = (np.log(default_value / current_value) + (alpha + beta * current_volatility) / 2) / np.sqrt(current_volatility)
                default_prob[j] += norm.sf(z_score)
            else:
                t_score = (np.log(default_value / current_value) + (alpha + beta * current_volatility) / 2) / np.sqrt(current_volatility / window_size)
                default_prob[j] += 1 - t.cdf(t_score, window_size - 1)
            date[j] = current_date

        # Normalize default probabilities for window
        default_prob[start_idx:end_idx] /= num_samples

    return pd.DataFrame({'date': date, 'default_prob': default_prob})