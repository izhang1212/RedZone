# convert raw rows into structured game states

import pandas as pd
from data.db import get_connection

# loads pbp data from sqlite into a pandas DataFrame
def get_cleaned_pbp_df():
    conn = get_connection()
    df = pd.read_sql('SELECT * FROM pbp_data', conn)
    conn.close()
    return df

def get_game_states(game_id):
    df = get_cleaned_pbp_df()
    return df[df['game_id'] == game_id].sort_values('play_id')