# runs simluations and returns final prob

import numpy as np
from models.brownian_bridge import simulate_bridge_paths

# runs simulation and returns the mean win prob
def run_monte_carlo(current_wp, time_remaning, num_sims = 10000):
    if current_wp >= 1.0:
        return 1.0
    if current_wp <= 0.0:
        return 0.0
    if time_remaning <= 0:
        return 1.0 if current_wp > 0.5 else 0.0
    
    outcomes = simulate_bridge_paths(current_wp, time_remaning, num_paths = num_sims)

    return np.mean(outcomes)