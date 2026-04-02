# Uses the square root of time scaling (σ√Δ) to simulate the expansion of uncertainty as we move toward the end of the game

import numpy as np

# simultes N paths from curr prob to end of the game
    # returns np.ndarray, an array of final outcomes (0,1) for each path
def simulate_bridge_paths(current_wp, time_remaning, num_paths = 10000, volatility = 1.0,):
    if time_remaning <= 0:
        return np.array([1.0 if current_wp >= 0.5 else 0.0] * num_paths)
    
    # represents step from curr time to end of game
    dt = time_remaning / 3600.0\
    
    # standard normal draws
        # generate random normal noise for end of the paths
    snd = np.random.standard_normal(num_paths)

    # Calculate the diffusion at the end of the time period
        # Scaling by sqrt(dt) follows the property of Brownian motion
    end_values = current_wp + (volatility * np.sqrt(dt) * snd)

    # values > 0.5 are wins, <= 0.5 are losses
        # this is the 'unconstrained' projection
    outcomes = (end_values > 0.5).astype(float)

    return outcomes