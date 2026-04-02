# runs simluations and returns final prob

import numpy as np
from models.brownian_bridge import simulate_bridge_paths
from models.drift import calculate_drift, build_drift_table, lookup_empirical_wp

def run_monte_carlo(game_state, num_sims = 10000, num_steps = 20):

    time_left = game_state.get('game_seconds_remaining', 0)
    home_score_diff = game_state.get('home_score_diff', 0)

    # game is over
    if time_left <= 0:
        wp = 1.0 if home_score_diff > 0 else (0.5 if home_score_diff == 0 else 0.0)
        return {
            'win_prob': wp,
            'paths': None,
            'drift_wp': wp,
            'model_wp': wp,
        }

    # get drift-adjusted starting point from empirical table
    drift_logit, drift_wp = calculate_drift(game_state)

    # also get the raw nflfastr wp for comparison
    raw_wp = game_state.get('home_wp', 0.5)

    # blend: weight empirical drift WP with nflfastr's WP
    # early in game, trust the empirical table more (larger sample sizes per bin)
    # late in game, the current wp already reflects game flow well
    time_pct = time_left / 3600.0
    empirical_weight = 0.6 + 0.2 * time_pct  # 0.6 at end, 0.8 at start
    blended_wp = empirical_weight * drift_wp + (1 - empirical_weight) * raw_wp
    blended_wp = np.clip(blended_wp, 0.001, 0.999)

    # incorporate pregame spread as a prior (weak pull toward pregame expectation)
    spread_line = game_state.get('spread_line', None)
    if spread_line is not None:
        from models.pregame import spread_to_win_prob
        pregame_wp = spread_to_win_prob(spread_line)
        # prior fades as game progresses — strong at kickoff, negligible by Q4
        prior_weight = 0.15 * time_pct  # max 15% at kickoff, 0% at game end
        blended_wp = (1 - prior_weight) * blended_wp + prior_weight * pregame_wp
        blended_wp = np.clip(blended_wp, 0.001, 0.999)

    # run simulation
    outcomes, paths = simulate_bridge_paths(
        current_wp=blended_wp,
        time_remaining=time_left,
        game_state=game_state,
        num_paths=num_sims,
        num_steps=num_steps,
    )

    model_wp = float(np.mean(outcomes))

    return {
        'win_prob': model_wp,
        'paths': paths,
        'drift_wp': drift_wp,
        'blended_input_wp': blended_wp,
        'model_wp': model_wp,
        'raw_nflfastr_wp': raw_wp,
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