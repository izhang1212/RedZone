# Init sqlite db

import sqlite3
from config import DB_PATH

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_DB():
    conn = get_connection()
    cursor = conn.cursor()

    # table for historical pbp
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pbp_data (
            game_id TEXT,
            play_id INTEGER,
            game_seconds_remaining INTEGER,
            score_differential INTEGER,
            down INTEGER,
            ydstogo INTEGER,
            yardline_100 INTEGER,
            posteam_timeouts_remaining INTEGER,
            defteam_timeouts_remaining INTEGER,
            wp REAL,
            PRIMARY KEY (game_id, play_id)
        )
    ''')
    
    # table for bankroll tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bankroll (
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            balance REAL,
            event TEXT
        )
    ''')

    conn.commit()
    conn.close()