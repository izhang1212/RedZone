# drift function - looks at current game state and determines directional force acting on the bridge

import numpy as np
import pandas as pd
from config import TOTAL_GAME_SECONDS

drift_table = None

# bin score differential by number of possessions needed
    # e.g. score diff of 8 needs at least 1 pos (one td + 2pt), while score diff of 9 needs at least 2 pos
def score_to_possession_bin(score_diff):
    abs_poss = min_possessions_needed(abs(score_diff))
    # cap at 5 for blowouts
    abs_poss = min(abs_poss, 5)
    return abs_poss if score_diff >= 0 else -abs_poss

# finds min number of scoring possessions to cover a deficit
    # uses fact that scores come in 2, 3, 6, 7, 8
def min_possessions_needed(score_deficit):
    if score_deficit is None or np.isnan(score_deficit):
        return 0
    if score_deficit <= 0:
        return 0
    return int(np.ceil(score_deficit / 8))

# Bin field position by scoring threat level
    # 0-19 -> 0 (redzone), 20-39 -> 1 (opponent territory), 40-59 -> 2 (midfield), 60-79 -> 3 (own territory), 80-100 -> 4 (pinned deep)
def field_position_bin(yardline_100):
    if yardline_100 is None or np.isnan(yardline_100):
        return 2
    return min(int(yardline_100 // 20), 4)

TIME_EDGES = np.array([1, 2, 5, 10, 15, 20, 25, 30, 37, 45, 52, 61])
# bin tiem into game phase
    # minutes matter more towards the end of the game
def time_to_bin(time_remaining):
    minutes = time_remaining / 60.0
    return int(np.searchsorted(TIME_EDGES, minutes))

# build drift lookup from historical pbp
    # group game states into bins (score, time, field pos) and computes actual win rate of each bin
def build_drift_table(pbp_df):
    global drift_table

    df = pbp_df.copy()

    if 'home_score_diff' not in df.columns:
        away_mask = df['posteam'] == df['away_team']
        df['home_score_diff'] = df['score_differential']
        df.loc[away_mask, 'home_score_diff'] = -df.loc[away_mask, 'score_differential']

    if 'home_win' not in df.columns:
        df['home_win'] = (df['result'] > 0).astype(float)
        df.loc[df['result'] == 0, 'home_win'] = 0.5

    df['home_has_ball'] = (df['posteam'] == df['home_team']).astype(int)

    df['home_field_pos'] = df['yardline_100']
    away_ball = df['home_has_ball'] == 0
    df.loc[away_ball, 'home_field_pos'] = 100 - df.loc[away_ball, 'yardline_100']

    df['score_bin'] = df['home_score_diff'].apply(score_to_possession_bin)
    df['time_bin'] = df['game_seconds_remaining'].apply(time_to_bin)
    df['field_bin'] = df['home_field_pos'].apply(field_position_bin)

    grouped = df.groupby(['score_bin', 'time_bin', 'field_bin', 'home_has_ball']).agg(
        win_rate=('home_win', 'mean'),
        count=('home_win', 'count')
    ).reset_index()

    fallback = df.groupby(['score_bin', 'time_bin']).agg(
        fallback_wr=('home_win', 'mean')
    ).reset_index()

    grouped = grouped.merge(fallback, on=['score_bin', 'time_bin'], how='left')

    min_samples = 30
    low_count = grouped['count'] < min_samples
    weight = np.minimum(grouped['count'] / min_samples, 1.0)
    grouped['win_rate'] = weight * grouped['win_rate'] + (1 - weight) * grouped['fallback_wr']

    drift_table = grouped.set_index(
        ['score_bin', 'time_bin', 'field_bin', 'home_has_ball']
    )[['win_rate', 'count']]

    return drift_table

# look up emperical wp for a game state
    # falls back if exact bin dne
def lookup_empirical_wp(score_diff, time_remaining, yardline_100, home_has_ball):
    global drift_table

    if drift_table is None:
        return 0.5
    
    score_bin = score_to_possession_bin(score_diff)
    time_bin = time_to_bin(time_remaining)
    field_bin = field_position_bin(yardline_100)
    has_ball = int(home_has_ball)

    key = (score_bin, time_bin, field_bin, has_ball)
    if key in drift_table.index:
        return drift_table.loc[key, 'win_rate']
    
    # fallback 1 - ignore field pos
    for fb in range(5):
        fallback_key = (score_bin, time_bin, fb, has_ball)
        if fallback_key in drift_table.index:
            return drift_table.loc[fallback_key, 'win_rate']
        
    # fallback 2 - ignore field pos and possession
    for fb in range(5):
        for hb in [0,1]:
            fallback_key = (score_bin, time_bin, fb, hb)
            if fallback_key in drift_table.index:
                return drift_table.loc[fallback_key, 'win_rate']

    return 0.5

# calc drift for the brownian bridge from the current game states
    # returns a value in logit space that shifts the bridge's center
    # (+) = trending toward home win, (-) = tranding toward home loss
def calculate_drift(game_state):
    score_diff = game_state.get('home_score_diff', 0)
    time_remaining = game_state.get('game_seconds_remaining', 0)
    yardline = game_state.get('yardline_100', 50)
    home_has_ball = game_state.get('posteam', '') == game_state.get('home_team', '')
    timeouts_pos = game_state.get('posteam_timeouts_remaining', 3)
    timeouts_def = game_state.get('defteam_timeouts_remaining', 3)
    down = game_state.get('down', 1)
    ydstogo = game_state.get('ydstogo', 10)

    empirical_wp = lookup_empirical_wp(score_diff, time_remaining, yardline, home_has_ball)

    # timeout edge (small adjustment, more advantage late game)
    time_pct = time_remaining / TOTAL_GAME_SECONDS
    timeout_adj = 0.0
    if time_pct < 0.15:  # only matters in Q4
        timeout_diff = timeouts_pos - timeouts_def
        if home_has_ball:
            timeout_adj = timeout_diff * 0.008
        else:
            timeout_adj = -timeout_diff * 0.008

    # down/distance adjustment
    down_adj = 0.0
    if down == 4 and home_has_ball:
        if ydstogo > 5:
            down_adj = -0.03  # likely punt or turnover on downs
        elif ydstogo <= 2:
            down_adj = 0.01   # likely to convert or go for it
    elif down == 4 and not home_has_ball:
        if ydstogo > 5:
            down_adj = 0.03   # opponent likely punting
        elif ydstogo <= 2:
            down_adj = -0.01

    adjusted_wp = np.clip(empirical_wp + timeout_adj + down_adj, 0.001, 0.999)
    
    # guard against NaN
    if np.isnan(adjusted_wp):
        adjusted_wp = 0.5

    drift_logit = np.log(adjusted_wp / (1 - adjusted_wp))
    return drift_logit, adjusted_wp
