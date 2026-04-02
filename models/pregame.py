# handles pregame prob (underdog, favorite, 50/50)

import numpy as np
from scipy.stats import norm

# converts an nfl point spread into win 
    # use normal cumulative dist function (norm.cdf) to map spread to prob
def spread_to_win_prob(spread):
    if spread is None:
        return 0.5
    # win_prob = norm_cdf(-spread / (std_dev * sqrt(2)))
        # use 13.45 as standard devation of score margins since this is historical SD of NFL outcomes
    return norm.cdf(-spread / 13.45)

# extract closing spread from a game and returns the implied prob
def get_pregame_wp(game_id, pbp_df):
    game_data = pbp_df[pbp_df['game_id'] == game_id]
    if game_data.empty:
        return 0.5
    
    # positive spreadline means hometeam is underdog, negative means favorite
    closing_spread = game_data['spread_line'].iloc[0]
    return spread_to_win_prob(closing_spread)