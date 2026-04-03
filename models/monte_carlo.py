# runs simulations and returns final prob

import numpy as np
from models.brownian_bridge import simulate_bridge_paths
from models.drift import calculate_drift, build_drift_table, lookup_empirical_wp

def run_monte_carlo(game_state, num_sims=10000, num_steps=20):

    time_left = game_state.get('game_seconds_remaining', 0)
    home_score_diff = game_state.get('home_score_diff', 0)

    # game is over
    if time_left <= 0:
        wp = 1.0 if home_score_diff > 0 else (0.5 if home_score_diff == 0 else 0.0)
        return {
            'win_prob': wp,
            'paths': None,
            'drift_wp': wp,
            'blended_input_wp': wp,
            'model_wp': wp,
        }

    # get empirical WP from drift table
    drift_logit, drift_wp = calculate_drift(game_state)

    # get nflfastr wp
    raw_wp = game_state.get('home_wp', 0.5)

    # blend drift table with nflfastr for best of both
    # nflfastr captures play-level nuance (down/distance/formation)
    # drift table captures score/time/possession/field position empirically
    # trust nflfastr more late game (when play state matters most)
    time_pct = time_left / 3600.0
    nfl_weight = 0.35 - 0.10 * time_pct  # 0.25 at kickoff, 0.35 at game end
    base_wp = (1 - nfl_weight) * drift_wp + nfl_weight * raw_wp
    base_wp = float(np.clip(base_wp, 0.001, 0.999))

    # incorporate pregame spread as a fading prior
    spread_line = game_state.get('spread_line', None)
    if spread_line is not None and not np.isnan(spread_line):
        from models.pregame import spread_to_win_prob
        pregame_wp = spread_to_win_prob(spread_line)
        prior_weight = 0.10 * time_pct  # max 10% at kickoff, 0% at game end
        base_wp = (1 - prior_weight) * base_wp + prior_weight * pregame_wp
        base_wp = float(np.clip(base_wp, 0.001, 0.999))

    # run bridge simulation from blended base
    outcomes, paths = simulate_bridge_paths(
        current_wp=base_wp,
        time_remaining=time_left,
        game_state=game_state,
        num_paths=num_sims,
        num_steps=num_steps,
    )

    sim_wp = float(np.mean(outcomes))

    # final blend: mostly base, simulation as small adjustment
    # bridge captures uncertainty dynamics the static blend can't
    sim_weight = 0.10 + 0.05 * time_pct  # 10% at game end, 15% at kickoff
    model_wp = (1 - sim_weight) * base_wp + sim_weight * sim_wp
    model_wp = float(np.clip(model_wp, 0.001, 0.999))

    return {
        'win_prob': model_wp,
        'paths': paths,
        'drift_wp': drift_wp,
        'blended_input_wp': base_wp,
        'model_wp': model_wp,
        'raw_nflfastr_wp': raw_wp,
        'sim_wp': sim_wp,
    }


def run_game_simulation(game_states_df, num_sims=10000, num_steps=20):
    results = []

    for _, row in game_states_df.iterrows():
        game_state = row.to_dict()
        result = run_monte_carlo(game_state, num_sims=num_sims, num_steps=num_steps)
        result['play_id'] = row.get('play_id')
        result['game_seconds_remaining'] = row.get('game_seconds_remaining')
        results.append(result)

    return results