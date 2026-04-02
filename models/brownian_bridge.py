# Brownian bridge simulates N paths from curr game state to end of game

import numpy as np
from config import TOTAL_GAME_SECONDS

# volatility depending on game state
    # decreases as time runs out and with larger leads
    # spikes slightly in "clutch time" because teams take more risks (going for it on 4th and long)
def volatility_schedule(time_pct, score_diff_abs):
    # base vol (decays with time)
    base = 0.12 * np.sqrt(np.maximum(time_pct, 0.001))

    if score_diff_abs > 0:
        score_dampening = 1.0 / (1.0 + 0.04 * score_diff_abs)
    else:
        score_dampening = 1.0

    clutch_boost = 1.0
    if time_pct < 0.12 and score_diff_abs <= 8:
        clutch_boost = 1.3
    elif time_pct <= 0.04 and score_diff_abs <= 16:
        clutch_boost = 1.5
    
    return base * score_dampening * clutch_boost

# simulate N brownian bridge paths from current wp to game end
def simulate_bridge_paths(current_wp, time_remaining, game_state = None, num_paths = 10000, num_steps = 20):
    if time_remaining <= 0:
        outcome = 1.0 if current_wp >= 0.5 else 0.0
        outcomes = np.full(num_paths, outcome)
        paths = np.full((1, num_paths), outcome)
        return outcomes, paths
    
    # extract score diff for volatility calculation
    score_diff_abs = 0
    if game_state is not None:
        score_diff_abs = abs(game_state.get('home_score_diff', 0))

    # work in logit space so paths stay bounded (0, 1) after transform
        # logit(p) = log(p / (1-p))
    wp_clipped = np.clip(current_wp, 0.001, 0.999)
    logit_start = np.log(wp_clipped / (1 - wp_clipped))

    time_pcts = np.linspace(time_remaining / TOTAL_GAME_SECONDS, 0.0, num_steps + 1)
    dt_steps = np.diff(time_pcts)

    logit_paths = np.zeros((num_steps + 1, num_paths))
    logit_paths[0, :] = logit_start

    # simulate forward through each time step
    for i in range(num_steps):
        t_pct = time_pcts[i]
        dt = abs(dt_steps[i])  # magnitude of time step

        if dt < 1e-10:
            logit_paths[i + 1, :] = logit_paths[i, :]
            continue

        # convert current logit paths back to probability for volatility calc
        current_probs = 1.0 / (1.0 + np.exp(-logit_paths[i, :]))
        # use mean probability to estimate "effective score state"
        mean_prob = np.mean(current_probs)
        effective_score = score_diff_abs * (mean_prob - 0.5) / max(abs(current_wp - 0.5), 0.01)
        effective_score = min(abs(effective_score), score_diff_abs)

        sigma = volatility_schedule(t_pct, effective_score)

        # Brownian bridge pull: as time → 0, paths get pulled toward 0 or 1
        # The bridge term adds mean-reversion toward the nearest terminal value
        remaining_pct = t_pct  # fraction of game left
        if remaining_pct > 0.001:
            # pull toward nearest terminal: logit → +inf (win) or -inf (loss)
            # strength of pull increases as remaining_pct → 0
            bridge_pull = logit_paths[i, :] * (dt / remaining_pct) * 0.1
        else:
            bridge_pull = 0.0

        # random increment
        noise = np.random.standard_normal(num_paths)
        dW = sigma * np.sqrt(dt) * noise

        logit_paths[i + 1, :] = logit_paths[i, :] + dW - bridge_pull

    # convert final logit values to probabilities
    final_probs = 1.0 / (1.0 + np.exp(-logit_paths[-1, :]))

    # terminal condition: prob > 0.5 → win, else loss
    outcomes = (final_probs > 0.5).astype(float)

    # convert all paths to probability space for visualization
    prob_paths = 1.0 / (1.0 + np.exp(-logit_paths))

    return outcomes, prob_paths