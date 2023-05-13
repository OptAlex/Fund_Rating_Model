import pandas as pd
import math
from recombinator.block_bootstrap import moving_block_bootstrap
from recombinator.optimal_block_length import optimal_block_length
from help_functions.help_functions_default_prob import get_fund_dict


def bootstrap_returns(df, bootstrap_samples=1):
    """
    This functions creates a bootstrap subsample for the log returns of each fund.
    :param df: df with historical log returns.
    :param bootstrap_samples: number of bootstrap samples the function creates.
    :return: df with bootstrapped returns.
    """

    # Create an empty DataFrame to store the bootstrapped returns
    bootstrap_df = pd.DataFrame()

    # Get a dictionary of fund names and their corresponding returns
    fund_dict = get_fund_dict(df)

    for fund_name, returns in fund_dict.items():
        # Calculate optimal block length
        b_star = optimal_block_length(returns)
        b_star_cb = math.ceil(b_star[0].b_star_cb)

        # Perform bootstrapping of the data
        bootstrap_returns = moving_block_bootstrap(
            returns,
            block_length=b_star_cb,
            replications=bootstrap_samples,
            replace=True,
        )

        bootstrap_returns_reshape = bootstrap_returns.reshape(-1, 1)
        bootstrap_returns_df = pd.DataFrame(
            bootstrap_returns_reshape, columns=[f"{fund_name}"]
        )
        bootstrap_df = pd.concat([bootstrap_df, bootstrap_returns_df], axis=1)

    return bootstrap_df
