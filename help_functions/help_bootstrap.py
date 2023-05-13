import pandas as pd
import math
from recombinator.block_bootstrap import circular_block_bootstrap
from recombinator.optimal_block_length import optimal_block_length
from help_functions.help_functions_default_prob import get_fund_dict


def bootstrap_returns(
        df: pd.DataFrame,
        bootstrap_samples: int,
        bootstrap_days: int,
) -> pd.DataFrame:
    """
    This functions creates a bootstrap subsample for the log returns of each fund.
    :param df: df with historical log returns.
    :param bootstrap_samples: number of bootstrap samples the function creates.
    :param bootstrap_days: number of days each bootstrap sample has.
    :return: df with bootstrapped returns.
    """

    fund_dict = get_fund_dict(df)

    for fund_name, returns in fund_dict.items():
        # Calculate optimal block length
        b_star = optimal_block_length(returns)
        b_star_cb = math.ceil(b_star[0].b_star_cb)

        # Perform bootstrapping of the data
        bootstrap_returns = circular_block_bootstrap(
            returns,
            block_length=b_star_cb,
            replications=bootstrap_samples,
            sub_sample_length=bootstrap_days,
            replace=True,
        )

    return bootstrap_returns