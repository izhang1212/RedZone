# Evaluate model predictions: Brier score, log loss, calibration

import numpy as np
import pandas as pd


def brier_score(predictions, outcomes):
    preds = np.array(predictions)
    actual = np.array(outcomes)
    return float(np.mean((preds - actual) ** 2))


def log_loss(predictions, outcomes, eps=1e-10):
    preds = np.clip(np.array(predictions), eps, 1 - eps)
    actual = np.array(outcomes)
    return float(-np.mean(actual * np.log(preds) + (1 - actual) * np.log(1 - preds)))


def calibration_table(predictions, outcomes, n_bins=10):
    preds = np.array(predictions)
    actual = np.array(outcomes)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(preds, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        rows.append({
            'bin_center': (bin_edges[i] + bin_edges[i + 1]) / 2,
            'predicted': float(preds[mask].mean()),
            'actual': float(actual[mask].mean()),
            'count': int(mask.sum()),
            'error': float(actual[mask].mean() - preds[mask].mean()),
        })

    return pd.DataFrame(rows)


def evaluate_model(results_df):
    df = results_df[results_df['home_win'].isin([0.0, 1.0])].copy()

    return {
        'model_brier': brier_score(df['model_wp'], df['home_win']),
        'model_logloss': log_loss(df['model_wp'], df['home_win']),
        'model_calibration': calibration_table(df['model_wp'], df['home_win']),
        'nflfastr_brier': brier_score(df['nflfastr_wp'], df['home_win']),
        'nflfastr_logloss': log_loss(df['nflfastr_wp'], df['home_win']),
        'nflfastr_calibration': calibration_table(df['nflfastr_wp'], df['home_win']),
        'drift_brier': brier_score(df['drift_wp'], df['home_win']),
        'drift_logloss': log_loss(df['drift_wp'], df['home_win']),
        'n_plays': len(df),
    }


def evaluate_by_game_phase(results_df):
    df = results_df[results_df['home_win'].isin([0.0, 1.0])].copy()

    phases = [
        ('Q1', df['game_seconds_remaining'] > 2700),
        ('Q2', (df['game_seconds_remaining'] > 1800) & (df['game_seconds_remaining'] <= 2700)),
        ('Q3', (df['game_seconds_remaining'] > 900) & (df['game_seconds_remaining'] <= 1800)),
        ('Q4 early', (df['game_seconds_remaining'] > 300) & (df['game_seconds_remaining'] <= 900)),
        ('Q4 late', (df['game_seconds_remaining'] > 120) & (df['game_seconds_remaining'] <= 300)),
        ('Final 2min', df['game_seconds_remaining'] <= 120),
    ]

    rows = []
    for name, mask in phases:
        if mask.sum() == 0:
            continue
        subset = df[mask]
        rows.append({
            'phase': name,
            'model_brier': brier_score(subset['model_wp'], subset['home_win']),
            'model_logloss': log_loss(subset['model_wp'], subset['home_win']),
            'nflfastr_brier': brier_score(subset['nflfastr_wp'], subset['home_win']),
            'n_plays': len(subset),
        })

    return pd.DataFrame(rows)


def find_worst_predictions(results_df, n=20):
    df = results_df[results_df['home_win'].isin([0.0, 1.0])].copy()
    df['squared_error'] = (df['model_wp'] - df['home_win']) ** 2
    return df.nlargest(n, 'squared_error')