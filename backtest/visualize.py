# Visualization tools for backtesting results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_calibration(model_cal, nflfastr_cal=None, title="Calibration Plot"):
    """
    Plot predicted vs actual win rate.
    Perfect calibration = diagonal line.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    # our model
    ax.plot(model_cal['predicted'], model_cal['actual'],
            'o-', color='#2196F3', markersize=8, linewidth=2, label='PocketOdds')

    # nflfastR comparison
    if nflfastr_cal is not None:
        ax.plot(nflfastr_cal['predicted'], nflfastr_cal['actual'],
                's-', color='#FF9800', markersize=8, linewidth=2, label='nflfastR')

    ax.set_xlabel('Predicted Win Probability', fontsize=12)
    ax.set_ylabel('Actual Win Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_game_wp(game_results, game_id=None):
    """
    Plot win probability over time for a single game.
    Shows model WP, nflfastR WP, and the actual outcome.
    """
    if isinstance(game_results, pd.DataFrame):
        df = game_results
    else:
        df = pd.DataFrame(game_results)

    if game_id is not None:
        df = df[df['game_id'] == game_id]

    fig, ax = plt.subplots(figsize=(12, 6))

    # time axis: convert seconds remaining to minutes elapsed
    minutes_elapsed = (3600 - df['game_seconds_remaining']) / 60

    ax.plot(minutes_elapsed, df['model_wp'],
            '-', color='#2196F3', linewidth=2, label='PocketOdds', alpha=0.9)

    ax.plot(minutes_elapsed, df['nflfastr_wp'],
            '-', color='#FF9800', linewidth=2, label='nflfastR', alpha=0.7)

    # actual outcome line
    outcome = df['home_win'].iloc[0]
    ax.axhline(y=outcome, color='green' if outcome == 1 else 'red',
               linestyle=':', alpha=0.5, label=f"Outcome: {'Home Win' if outcome == 1 else 'Home Loss'}")

    # 50% reference
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.2)

    # quarter markers
    for q_min in [15, 30, 45]:
        ax.axvline(x=q_min, color='gray', linestyle='--', alpha=0.2)

    ax.set_xlabel('Minutes Elapsed', fontsize=12)
    ax.set_ylabel('Home Win Probability', fontsize=12)
    title = f"Win Probability: {df['game_id'].iloc[0]}" if 'game_id' in df.columns else "Win Probability"
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error_distribution(results_df):
    """
    Histogram of prediction errors (model_wp - home_win).
    Should be centered at 0 with thin tails.
    """
    df = results_df[results_df['home_win'].isin([0.0, 1.0])].copy()
    errors = df['model_wp'] - df['home_win']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # raw error distribution
    axes[0].hist(errors, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Prediction Error (Model WP - Outcome)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution')
    mean_err = errors.mean()
    axes[0].text(0.02, 0.95, f'Mean error: {mean_err:+.4f}',
                 transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # model vs nflfastR squared error comparison
    model_se = (df['model_wp'] - df['home_win']) ** 2
    nfl_se = (df['nflfastr_wp'] - df['home_win']) ** 2
    diff = model_se - nfl_se  # negative = our model is better

    axes[1].hist(diff, bins=50, color='#4CAF50', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Squared Error Difference (Ours - nflfastR)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Model vs nflfastR (left = we\'re better)')
    pct_better = (diff < 0).mean()
    axes[1].text(0.02, 0.95, f'We beat nflfastR on {pct_better:.1%} of plays',
                 transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_model_comparison_over_time(results_df):
    """
    Compare model vs nflfastR Brier score across game phases.
    Shows where our model is stronger and weaker.
    """
    df = results_df[results_df['home_win'].isin([0.0, 1.0])].copy()
    df['minutes_remaining'] = df['game_seconds_remaining'] / 60

    # bin into 5-minute windows
    bins = list(range(0, 65, 5))
    labels = [f"{b}-{b+5}" for b in bins[:-1]]
    df['time_bin'] = pd.cut(df['minutes_remaining'], bins=bins, labels=labels, right=False)

    model_brier = df.groupby('time_bin', observed=True).apply(
        lambda x: np.mean((x['model_wp'] - x['home_win']) ** 2)
    )
    nfl_brier = df.groupby('time_bin', observed=True).apply(
        lambda x: np.mean((x['nflfastr_wp'] - x['home_win']) ** 2)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(model_brier))
    ax.bar([i - 0.15 for i in x], model_brier.values, 0.3,
           label='PocketOdds', color='#2196F3', alpha=0.8)
    ax.bar([i + 0.15 for i in x], nfl_brier.values, 0.3,
           label='nflfastR', color='#FF9800', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_brier.index, rotation=45, ha='right')
    ax.set_xlabel('Minutes Remaining')
    ax.set_ylabel('Brier Score (lower = better)')
    ax.set_title('Model Accuracy by Game Phase')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig