"""Shared feature-engineering functions for the Aido Rover anomaly-detection pipeline.

Used by W02_Preprocessing_Pipeline.ipynb (tabular 40-D matrix), W02_Sequence_and_RL_Scaffolding.ipynb
(window-tensor physical channels), and W03_Neural_Network_Baseline.ipynb (MLP's tabular matrix) — a
single definition point so the FFT and physical-feature formulas can't drift between notebooks.
"""
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

WINDOW = 50
FFT_CHANNELS = ['torque_0', 'torque_1', 'torque_2', 'torque_3', 'lidar_distance']
FFT_FEATURE_NAMES = ['dom_freq', 'centroid', 'bandwidth', 'total_power', 'peak_to_mean']
MODEL_SENSOR_COLS = ['gps_dlat', 'gps_dlon', 'lidar_distance', 'battery_soc',
                     'torque_0', 'torque_1', 'torque_2', 'torque_3', 'ambient_temp']
EPS_DISP = 1e-3   # fixed physical floor for stall_ratio, not fit from data


def add_gps_deltas(df):
    """Return a copy of df with gps_dlat/gps_dlon per-step delta columns added.

    Absolute position is a fixed-map shortcut to "which spot on the loop", not a transferable
    fault signal; the slip/stuck mechanism is causally a displacement collapse, which the delta
    captures directly.
    """
    df = df.copy()
    df['gps_dlat'] = df['gps_lat'].diff().fillna(0.0)
    df['gps_dlon'] = df['gps_lon'].diff().fillna(0.0)
    return df


def fft_features(seg, fs=10.0):
    """One window of a single channel -> 5 spectral features."""
    freqs = rfftfreq(len(seg), d=1 / fs)
    mag = np.abs(rfft(seg))
    mag[0] = 0
    dom_freq = freqs[np.argmax(mag)]
    centroid = np.dot(freqs, mag) / np.sum(mag)
    bandwidth = np.sqrt(np.dot((freqs - centroid) ** 2, mag) / np.sum(mag))
    total_power = np.dot(mag, mag)
    peak_to_mean = np.max(mag) / np.mean(mag)
    return [dom_freq, centroid, bandwidth, total_power, peak_to_mean]


def physical_channels(df):
    """Per-step cross-channel physical features (instantaneous, no rolling stats).

    Both fault mechanisms are defined by a relationship *between* channels that no per-channel
    view (FFT included) can express: slip is one wheel's torque diverging from the other three;
    stuck is torque staying high while displacement collapses toward zero.

    df must already have gps_dlat/gps_dlon (see add_gps_deltas).
    Returns (inter_wheel_std, stall_ratio), each shape (len(df),).
    """
    tq = df[['torque_0', 'torque_1', 'torque_2', 'torque_3']].values
    inter_wheel_std = tq.std(axis=1)
    displacement = np.sqrt(df['gps_dlat'].values ** 2 + df['gps_dlon'].values ** 2)
    stall_ratio = tq.mean(axis=1) / (displacement + EPS_DISP)
    return inter_wheel_std, stall_ratio


def build_feature_matrix(df_clean, window=WINDOW):
    """Full 40-D tabular feature matrix for RF/MLP: 9 raw (GPS-delta) + 25 FFT + 6 physical
    (2 instantaneous + 4 rolling mean/max, matching the FFT window).

    Sequence models (CNN/RNN/Transformer) should use `physical_channels` directly as extra raw
    window channels instead — they perform their own temporal aggregation, so the rolling
    mean/max here would be redundant for them.

    Returns (feature_matrix, feature_names, row_idx_array, label_array).
    """
    df_clean = add_gps_deltas(df_clean)
    inter_wheel_std, stall_ratio = physical_channels(df_clean)

    phys_features = np.column_stack([
        inter_wheel_std, stall_ratio,
        pd.Series(inter_wheel_std).rolling(window).mean().values,
        pd.Series(inter_wheel_std).rolling(window).max().values,
        pd.Series(stall_ratio).rolling(window).mean().values,
        pd.Series(stall_ratio).rolling(window).max().values,
    ])
    phys_names = ['inter_wheel_std', 'stall_ratio', 'inter_wheel_std_roll_mean', 'inter_wheel_std_roll_max',
                  'stall_ratio_roll_mean', 'stall_ratio_roll_max']

    fft_names = [f'{ch}_{feat}' for ch in FFT_CHANNELS for feat in FFT_FEATURE_NAMES]
    feature_names = MODEL_SENSOR_COLS + fft_names + phys_names

    rows, label, row_idx_list = [], [], []
    for i in range(window, len(df_clean)):
        raw = df_clean[MODEL_SENSOR_COLS].iloc[i].values.tolist()
        fft = [v for ch in FFT_CHANNELS for v in fft_features(df_clean[ch].iloc[i - window:i].values)]
        rows.append(raw + fft + list(phys_features[i]))
        label.append(df_clean['anomaly_label'].iloc[i])
        row_idx_list.append(i)

    feature_matrix = np.array(rows)
    label_array = np.array(label)
    row_idx_array = np.array(row_idx_list)
    return feature_matrix, feature_names, row_idx_array, label_array
