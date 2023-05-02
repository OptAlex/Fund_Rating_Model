import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm

# Define sliding window size and step
window_size = 252
step = 1

# Define threshold for default
threshold = 0.1

# Define number of bootstrap samples
num_samples = 1

# Load data into DataFrame
df = pd.read_csv('~/programming/python/uni/projects/market_risk/dummy_data.csv').dropna().reset_index()
df[["return_agg", "return_spy", "return_mergaai"]] = df[["return_agg", "return_spy", "return_mergaai"]] * 100

# Define function to estimate default probability using GARCH and statistical method
def estimate_default_prob(df):
    # Initialize default probability array
    default_prob = np.zeros(len(df))

    # Iterate through sliding windows
    for i in range(0, len(df) - window_size, step):
        # Define window start and end indices
        start_idx = i
        end_idx = i + window_size

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
            else:
                current_value = df['price_agg'][j]

            default_value = start_value * (1 - threshold)
            z_score = (np.log(default_value / current_value) + (alpha + beta * current_volatility) / 2) / np.sqrt(
                current_volatility)
            default_prob[j] += norm.sf(z_score)

        # Normalize default probabilities for window
        default_prob[start_idx:end_idx] /= num_samples

    return default_prob


# Call function to estimate default probabilities
default_prob = estimate_default_prob(df)

# Print default probabilities
print(default_prob)
