import pandas as pd
from help_functions.estimate_default_prob import estimate_historical_stat_default_prob, estimate_historical_count_default_prob, estimate_mc_stat_default_prob, estimate_mc_count_default_prob
import seaborn as sns
import matplotlib.pyplot as plt

# Load data into DataFrame, these steps are only for the dummy data
df = (
    pd.read_csv("~/programming/python/uni/projects/market_risk/dummy_data.csv")
    .dropna()
    .reset_index()
    .drop("index", axis=1)
    .iloc[:300]
)

# Call function to estimate default probabilities
estimate_mc_count_default_prob = estimate_mc_count_default_prob(
    df=df, threshold=-0.005, num_samples=50, prediction_days=30
)
"""estimate_historical_count_default_prob = estimate_historical_count_default_prob(
    df=df, threshold=-0.01, window_size=30, step=1,
)
estimate_historical_stat_default_prob = estimate_historical_stat_default_prob(
    df=df, threshold=-0.1, window_size=30, step=1, score="t_score"
)
estimate_mc_stat_default_prob = estimate_mc_stat_default_prob(
    df=df, threshold=-0.03, num_samples=50, prediction_days=30, score="t_score"
)
estimate_mc_stat_default_prob = estimate_mc_count_default_prob(
    df=df, threshold=-0.005, num_samples=50, prediction_days=30
)"""

# Print default probabilities
print(estimate_mc_count_default_prob)


# create a figure with three boxplots
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=estimate_mc_count_default_prob, ax=ax, palette='PuBu_r', showfliers=True, width=0.3, whis=[5, 95])
ax.set_title('Default Probability Forecast (30 days) for Each Fund', fontsize=14)
ax.set_xlabel('Fund', fontsize=12)
ax.set_ylabel('Default Probability', fontsize=12)
ax.tick_params(labelsize=10)

# save the figure
fig.savefig('mc_forecast_boxplot_fund.png', dpi=300, bbox_inches='tight')

# create a figure with one boxplot
fig, ax = plt.subplots(figsize=(4, 4))
sns.boxplot(data=estimate_mc_count_default_prob.stack().reset_index(level=1, drop=True), ax=ax, color='lightsteelblue', showfliers=True, width=0.1, whis=[5, 95])
ax.set_title('Default Probability Forecast (30 days) for All Funds', fontsize=14)
ax.set_xlabel('All Funds', fontsize=12)
ax.set_ylabel('Default Probability', fontsize=12)
ax.tick_params(labelsize=10)

# save the figure
fig.savefig('mc_forecast_boxplot_all_funds.png', dpi=300, bbox_inches='tight')