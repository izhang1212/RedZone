from data.load_pbp import fetch_and_store_pbp
from data.preprocess import get_cleaned_pbp_df
from models.drift import build_drift_table
from backtest.replay import replay_season, replay_season_walkforward
from backtest.evaluate import evaluate_model, evaluate_by_game_phase, find_worst_predictions
from backtest.visualize import plot_calibration, plot_game_wp, plot_error_distribution, plot_model_comparison_over_time
import matplotlib.pyplot as plt

def init(fetch_if_empty=True):
    if fetch_if_empty:
        from data.db import get_connection
        conn = get_connection()
        try:
            count = conn.execute("SELECT COUNT(*) FROM pbp_data").fetchone()[0]
        except:
            count = 0
        conn.close()
        if count == 0:
            print("Database empty — fetching play-by-play data...")
            fetch_and_store_pbp()

    print("Loading play-by-play data...")
    pbp_df = get_cleaned_pbp_df()

    print("Building drift table...")
    build_drift_table(pbp_df)

    print("Ready.")
    return pbp_df

if __name__ == "__main__":
    pbp_df = init()

    # walk-forward: build drift from 2021-2022, test on 2023
    print("\nReplaying 10 games from 2023 (walk-forward)...")
    results = replay_season_walkforward(2023, pbp_df=pbp_df, num_sims=5000, num_steps=15, max_games=10)

    # metrics
    print("\n" + "=" * 60)
    metrics = evaluate_model(results)
    print(f"{'Metric':<25} {'Our Model':>12} {'nflfastR':>12} {'Drift Only':>12}")
    print("-" * 60)
    print(f"{'Brier Score':<25} {metrics['model_brier']:>12.4f} {metrics['nflfastr_brier']:>12.4f} {metrics['drift_brier']:>12.4f}")
    print(f"{'Log Loss':<25} {metrics['model_logloss']:>12.4f} {metrics['nflfastr_logloss']:>12.4f} {metrics['drift_logloss']:>12.4f}")
    print(f"{'Plays evaluated':<25} {metrics['n_plays']:>12,}")

    # calibration
    print("\nCalibration:")
    cal = metrics['model_calibration']
    print(f"{'Bin':>8} {'Predicted':>10} {'Actual':>10} {'Count':>8} {'Error':>8}")
    print("-" * 48)
    for _, row in cal.iterrows():
        print(f"{row['bin_center']:>8.1%} {row['predicted']:>10.3f} "
              f"{row['actual']:>10.3f} {row['count']:>8,} {row['error']:>+8.3f}")

    # phase breakdown
    print("\nAccuracy by game phase:")
    phases = evaluate_by_game_phase(results)
    print(f"{'Phase':<15} {'Brier':>8} {'LogLoss':>8} {'nflfast':>8} {'Plays':>8}")
    print("-" * 50)
    for _, row in phases.iterrows():
        print(f"{row['phase']:<15} {row['model_brier']:>8.4f} {row['model_logloss']:>8.4f} "
              f"{row['nflfastr_brier']:>8.4f} {row['n_plays']:>8,}")

    # worst predictions
    print("\nWorst predictions:")
    worst = find_worst_predictions(results, n=10)
    print(f"{'Game':<25} {'Time':>6} {'Score':>6} {'Model':>7} {'Actual':>7}")
    print("-" * 55)
    for _, row in worst.iterrows():
        mins = row['game_seconds_remaining'] / 60
        print(f"{row['game_id']:<25} {mins:>5.1f}m {row['home_score_diff']:>+5.0f} "
              f"{row['model_wp']:>7.3f} {row['home_win']:>7.0f}")

    # plots
    fig1 = plot_calibration(metrics['model_calibration'], metrics['nflfastr_calibration'])
    fig2 = plot_error_distribution(results)
    fig3 = plot_model_comparison_over_time(results)

    sample_game = results['game_id'].unique()[0]
    fig4 = plot_game_wp(results, game_id=sample_game)

    plt.show()