# Replay historical games through the model and record predictions

import pandas as pd
import numpy as np
from tqdm import tqdm

from data.preprocess import get_cleaned_pbp_df
from models.monte_carlo import run_monte_carlo
from models.drift import build_drift_table


def replay_game(game_df, num_sims=5000, num_steps=15):
    results = []

    for _, row in game_df.iterrows():
        game_state = row.to_dict()
        mc_result = run_monte_carlo(game_state, num_sims=num_sims, num_steps=num_steps)

        results.append({
            'game_id': row['game_id'],
            'play_id': row['play_id'],
            'game_seconds_remaining': row['game_seconds_remaining'],
            'home_score_diff': row.get('home_score_diff', 0),
            'model_wp': mc_result['model_wp'],
            'nflfastr_wp': row.get('home_wp', 0.5),
            'drift_wp': mc_result['drift_wp'],
            'blended_input_wp': mc_result['blended_input_wp'],
            'home_win': row['home_win'],
        })

    return results


def replay_season(season, pbp_df=None, num_sims=5000, num_steps=15, max_games=None):
    if pbp_df is None:
        pbp_df = get_cleaned_pbp_df()

    season_df = pbp_df[pbp_df['season'] == season].copy()
    game_ids = season_df['game_id'].unique()

    if max_games is not None:
        game_ids = game_ids[:max_games]

    all_results = []

    for game_id in tqdm(game_ids, desc=f"Season {season}"):
        game_df = season_df[season_df['game_id'] == game_id].sort_values('play_id')
        game_results = replay_game(game_df, num_sims=num_sims, num_steps=num_steps)
        all_results.extend(game_results)

    return pd.DataFrame(all_results)


def replay_all_seasons(seasons, pbp_df=None, num_sims=5000, num_steps=15):
    if pbp_df is None:
        pbp_df = get_cleaned_pbp_df()

    build_drift_table(pbp_df)

    all_dfs = []
    for season in seasons:
        season_df = replay_season(season, pbp_df=pbp_df,
                                  num_sims=num_sims, num_steps=num_steps)
        all_dfs.append(season_df)

    return pd.concat(all_dfs, ignore_index=True)

# replay a season using only prior seasons to build the drift table
def replay_season_walkforward(season, pbp_df=None, num_sims=5000, num_steps=15, max_games=None):
    if pbp_df is None:
        pbp_df = get_cleaned_pbp_df()

    # build drift table from seasons BEFORE the test season only
    prior = pbp_df[pbp_df['season'] < season]
    if len(prior) == 0:
        # no prior data, fall back to all data
        build_drift_table(pbp_df)
    else:
        build_drift_table(prior)

    return replay_season(season, pbp_df=pbp_df, num_sims=num_sims,
                         num_steps=num_steps, max_games=max_games)