from models.monte_carlo import run_monte_carlo
from config import NUM_SIMULATIONS

def main():
    print("--- RedZone System: Phase 3 ---")
    
    # Scenario: 4th Quarter, 5 minutes left, Team A is slightly favored
    current_wp = 0.60
    time_left = 300 
    
    print(f"[1/1] Running Monte Carlo Simulation...")
    print(f"  > Current WP: {current_wp:.2%}")
    print(f"  > Time Remaining: {time_left}s")
    print(f"  > Simulations: {NUM_SIMULATIONS}")
    
    simulated_wp = run_monte_carlo(current_wp, time_left, num_sims=NUM_SIMULATIONS)
    
    print(f"\nResulting Simulated Win Probability: {simulated_wp:.2%}")
    
    # Internal logic check
    edge = simulated_wp - current_wp
    print(f"Variance/Drift adjustment: {edge:+.2%}")

    print("\nPhase 3 Complete. Ready for Part 4: Drift Function.")

if __name__ == "__main__":
    main()