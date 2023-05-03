import pandas as pd
from help_functions.estimate_default_prob import estimate_default_prob

# Load data into DataFrame, these steps are only for the dummy data
df = (
    pd.read_csv("~/programming/python/uni/projects/market_risk/dummy_data.csv")
    .dropna()
    .reset_index()
    .drop("index", axis=1)
    .iloc[:300]
)
df[["return_agg", "return_spy", "return_mergaai"]] = (
    df[["return_agg", "return_spy", "return_mergaai"]] * 100
)

# Call function to estimate default probabilities
default_prob = estimate_default_prob(
    df=df, threshold=0.1, num_samples=1, window_size=252, step=1, score="t_score"
)

# Print default probabilities
print(default_prob)

