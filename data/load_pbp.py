import nfl_data_py as nfl
import pandas as pd
from data.db import get_connection
from config import TERMS

# download pbp data and save it to sqlite
def fetch_and_store():
    # fetch data for seasons defined in config.py
    df = nfl.import_p_p_data(TERMS["seasons"])

    # define only cols we need
    cols = [
        'game_id', 'play_id', 'game_seconds_remaining', 'score_differential',
        'down', 'ydstogo', 'yardline_100', 'posteam_timeouts_remaining',
        'defteam_timeouts_remaining', 'wp'
    ]

    # drop rows with missing info
    df_clean = df[cols].dropna()

    conn = get_connection()
    
    # save and close conn
    df_clean.to_sql('pbp_data', conn, if_exists = 'replace', index = False)
    conn.close()

    return len(df_clean)