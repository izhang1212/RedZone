# convert raw rows into structured game states

import pandas as pd
import numpy as np
from data.db import get_connection

# loads pbp data from sqlite into a pandas DataFrame
def get_cleaned_pbp_df():
    conn = get_connection()
    df = pd.read_sql('SELECT * FROM pbp_data', conn)
    conn.close()

    # normalize wp to home team's perspective
    df['home_wp'] = np.where(
        df['posteam'] == df['home_team'],
        df['wp'],
        1 - df['wp']
    )

    # time remaining percentage [0,1]
    df['time_remaining_pct'] = df['game_seconds_remaining'] / 3600.0

    # score differential from home team's perspective
    df['home_score_diff'] = np.where(
        df['posteam'] == df['home_team'],
        df['score_differential'],
        -df['score_differential']
    )

    # drop rows with missing derived fields
    df = df.dropna(subset=['score_differential', 'result', 'wp'])

    # binary outcome: did the home team win?
    # result = home_score - away_score (positive = home win)
    df['home_win'] = np.where(df['result'] > 0, 1.0, 0.0)
    df.loc[df['result'] == 0, 'home_win'] = 0.5  # ties

    return df

def get_game_states(game_id):
    df = get_cleaned_pbp_df()
    return df[df['game_id'] == game_id].sort_values('play_id')