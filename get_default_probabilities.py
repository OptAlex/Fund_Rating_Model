import pandas as pd
from help_functions.estimate_cumulative_default_prob import estimate_default_prob, plot_default_prob

# Load data into DataFrame, these steps are only for the dummy data
df = (
    pd.read_csv("~/programming/python/uni/projects/market_risk/dummy_data.csv")
    .dropna()
    .reset_index()
    .drop("index", axis=1)
)

# Call function to estimate default probabilities
default_prob = estimate_default_prob(
    df=df, threshold=-0.15, bootstrap_samples=50, simulation_samples=100, prediction_days=252
)

# Print default probabilities
print(default_prob)

# Plot the histogram of the default probabilities
plot_default_prob(default_prob, confidence_level=0.95)