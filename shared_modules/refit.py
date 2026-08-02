"""Deterministic refits of the Week 2-4 models, with persisted checkpoints.

Weeks 2-4 trained every model inside a notebook and kept only the printed metrics, so the fitted
objects no longer exist by the time Week 7 needs them for SHAP, attention extraction, quantisation
and re-measured latency. Each ``refit_*`` here re-runs the original notebook's training path —
same architecture, optimiser, seed, split handling, scaler fitting, early stopping and threshold
tuning — and ``save_all`` writes the fitted objects under ``saved_models/`` together with a
manifest that records each published target metric next to the achieved one.

Source notebooks: ``W02_RF_Benchmark.ipynb`` (Rover RF); ``W03_Neural_Network_Baseline.ipynb``
(Rover MLP, Rover 1D-CNN, Fari RF, Fari MLP); ``W03_Sequence_Models_RNN_vs_Transformer.ipynb``
(LSTM, GRU, Transformer); ``W04_Trajectory_Prediction.ipynb`` (trajectory regressors).

``python -m shared_modules.refit`` refits everything, saves it and prints the verification table.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared_modules.features import build_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
TEMP_DIR = DATA_DIR / 'temp'
SAVED_DIR = ROOT / 'saved_models'
MANIFEST = SAVED_DIR / 'manifest.json'

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cuDNN's default Conv1d backward accumulates gradients atomically, which alone makes the 1D-CNN
# refit vary from run to run (test F1 spread 0.683-0.738 over 8 draws, while its AUC holds at
# 0.9585 +/- 0.0010). Restricting cuDNN to deterministic algorithms pins that down; every other
# model here still reproduces its published value bit-exactly under the restriction.
torch.backends.cudnn.deterministic = True

WINDOW = 50            # canonical sequence-model window (W03 §2)
N_HIST, N_FUT = 10, 5  # trajectory history / horizon lengths (W04)
DT = 0.1               # s, 10 Hz planner tick (W04)

# Published canonical-fold / canonical-seed values from the source notebooks. `save_all` compares
# each refit against these and records the delta; nothing here is tuned to make them match.
TARGETS = {
    ('rover', 'rf'): {'threshold': 0.49, 'test_f1': 0.7359, 'test_auc': 0.9668},
    ('rover', 'mlp'): {'threshold': 0.76, 'test_f1': 0.7358, 'test_auc': 0.9677},
    ('rover', 'cnn'): {'threshold': 0.78, 'test_f1': 0.7317, 'test_auc': 0.9585},
    ('rover', 'lstm'): {'threshold': 0.49, 'test_f1': 0.6739, 'test_auc': 0.9610},
    ('rover', 'gru'): {'threshold': 0.65, 'test_f1': 0.7423, 'test_auc': 0.9733},
    ('rover', 'transformer'): {'threshold': 0.87, 'test_f1': 0.7782, 'test_auc': 0.9861},
    ('fari', 'rf'): {'threshold': 0.43, 'test_f1': 0.7559, 'test_auc': 0.8077},
    ('fari', 'mlp'): {'threshold': 0.40, 'test_f1': 0.7555, 'test_auc': 0.8229},
    ('trajectory', 'cv'): {'mean_cm': 2.768},
    ('trajectory', 'linear'): {'mean_cm': 1.504},
    ('trajectory', 'mlp'): {'mean_cm': 1.61},
    ('trajectory', 'lstm'): {'mean_cm': 1.45},
    ('trajectory', 'transformer'): {'mean_cm': 1.79},
}


# --- shared training helpers (ported from the W03 notebooks) ------------------------------
def class_weight_tensor(y, device):
    classes, counts = np.unique(y, return_counts=True)
    w = counts.sum() / (len(classes) * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def make_loader(X, y, batch_size=128, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _eval_split(model, X, y, criterion, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.long, device=device)
        out = model(Xt)
        loss = criterion(out, yt).item()
        f1 = f1_score(y, out.argmax(1).cpu().numpy())
    return loss, f1


def train_with_early_stopping(model, train_loader, X_tr, y_tr, X_va, y_va, criterion, optimizer,
                              device, max_epochs=100, patience=10):
    model.to(device)
    best_val_loss = np.inf
    best_state = None
    best_epoch = -1
    n_bad = 0
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    n_epochs_run = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        train_loss, train_f1 = _eval_split(model, X_tr, y_tr, criterion, device)
        val_loss, val_f1 = _eval_split(model, X_va, y_va, criterion, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        n_epochs_run = epoch + 1

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            n_bad = 0
        else:
            n_bad += 1
            if n_bad >= patience:
                break

    model.load_state_dict(best_state)
    history['best_epoch'] = best_epoch
    return model, history, n_epochs_run


def tune_threshold(y_val, proba_val, grid=None):
    """Val-tuned decision threshold (max anomaly-F1 on val), applied unchanged to test."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_val, (proba_val >= t).astype(int)) for t in grid]
    best_i = int(np.argmax(f1s))
    return float(grid[best_i]), float(f1s[best_i])


def predict_proba(model, X, device):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.softmax(out, dim=1)[:, 1].cpu().numpy()


def _classification_metrics(y_te, proba_te, threshold):
    pred = (proba_te >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return {
        'test_f1': float(f1_score(y_te, pred)),
        'test_f1_default': float(f1_score(y_te, (proba_te >= 0.5).astype(int))),
        'test_auc': float(roc_auc_score(y_te, proba_te)),
        'test_precision': float(tp / (tp + fp)) if (tp + fp) else 0.0,
        'test_recall': float(tp / (tp + fn)) if (tp + fn) else 0.0,
        'confusion_matrix': [int(tn), int(fp), int(fn), int(tp)],
    }


# --- model modules (ported verbatim from the source notebooks) ----------------------------
class MLP(nn.Module):
    """W03 tabular classifier: in_dim -> hidden... -> 2."""

    def __init__(self, in_dim=40, hidden=(64, 32), activation='relu', dropout=0.3):
        super().__init__()
        act_layer = nn.ReLU if activation == 'relu' else (lambda: nn.LeakyReLU(0.01))
        dims = [in_dim] + list(hidden)
        layers = []
        for i in range(len(hidden)):
            layers += [nn.Linear(dims[i], dims[i + 1]), act_layer(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """W03 windowed classifier; `pool` selects the head over the time axis."""

    def __init__(self, in_channels=11, dropout=0.3, pool='avg'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = pool
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        fc_in = 128 if pool == 'concat' else 64
        self.fc = nn.Sequential(nn.Linear(fc_in, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):
        x = self.conv(x)
        avg_p = self.gap(x).squeeze(-1)
        max_p = self.gmp(x).squeeze(-1)
        if self.pool == 'avg':
            pooled = avg_p
        elif self.pool == 'max':
            pooled = max_p
        else:
            pooled = torch.cat([avg_p, max_p], dim=1)
        return self.fc(pooled)


class RNNClassifier(nn.Module):
    """W03 many-to-one recurrent classifier reading the final timestep's hidden state."""

    def __init__(self, input_size=11, hidden_size=64, rnn_type='lstm', dropout=0.3):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size

    def forward(self, x, return_hidden_seq=False):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        logits = self.fc(self.dropout(last))
        if return_hidden_seq:
            return logits, out
        return logits


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (max_len 50 for W03 windows, 10 for W04 history)."""

    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    """W03 encoder-only classifier, mean-pooled over the window."""

    def __init__(self, input_size=11, d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                 dropout=0.1, max_len=50):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 2)
        self.d_model = d_model

    def forward(self, x):
        h = self.pos_enc(self.input_proj(x))
        h = self.transformer_encoder(h)
        pooled = h.mean(dim=1)
        return self.fc(self.dropout(pooled))


class MLPRegressor(nn.Module):
    """W04 trajectory regressor: flattened 10x8 history -> 5 waypoints."""

    def __init__(self, n_hist=10, n_feat=8, n_fut=5, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_hist * n_feat, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_fut * 2))
        self.n_fut = n_fut

    def forward(self, x):
        return self.net(x).view(-1, self.n_fut, 2)


class LSTMRegressor(nn.Module):
    """W04 trajectory regressor: final hidden state -> all 5 waypoints jointly."""

    def __init__(self, n_feat=8, hidden=64, n_fut=5):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden, n_fut * 2)
        self.n_fut = n_fut

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :])).view(-1, self.n_fut, 2)


class TrajTransformer(nn.Module):
    """W04 seq2seq regressor: encoder over the history, 5 learned horizon queries cross-attending."""

    def __init__(self, n_feat=8, d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                 dropout=0.1, n_hist=10, n_fut=5):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=n_hist)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.horizon_query = nn.Parameter(torch.randn(n_fut, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x, return_attn=False):
        mem = self.encoder(self.pos_enc(self.input_proj(x)))
        q = self.horizon_query.unsqueeze(0).expand(x.size(0), -1, -1)
        ctx, attn = self.cross_attn(q, mem, mem, need_weights=True)
        out = self.head(self.norm(ctx))
        if return_attn:
            return out, attn
        return out


TORCH_CLASSES = {
    ('rover', 'mlp'): MLP, ('rover', 'cnn'): CNN1D,
    ('rover', 'lstm'): RNNClassifier, ('rover', 'gru'): RNNClassifier,
    ('rover', 'transformer'): TransformerClassifier,
    ('fari', 'mlp'): MLP,
    ('trajectory', 'mlp'): MLPRegressor, ('trajectory', 'lstm'): LSTMRegressor,
    ('trajectory', 'transformer'): TrajTransformer,
}
# constructor keys per class, so `load_model` can rebuild from the saved config dict
_INIT_KEYS = {
    MLP: ('in_dim', 'hidden', 'activation', 'dropout'),
    CNN1D: ('in_channels', 'dropout', 'pool'),
    RNNClassifier: ('input_size', 'hidden_size', 'rnn_type', 'dropout'),
    TransformerClassifier: ('input_size', 'd_model', 'nhead', 'num_layers', 'dim_feedforward',
                            'dropout', 'max_len'),
    MLPRegressor: ('n_hist', 'n_feat', 'n_fut', 'dropout'),
    LSTMRegressor: ('n_feat', 'hidden', 'n_fut'),
    TrajTransformer: ('n_feat', 'd_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout',
                      'n_hist', 'n_fut'),
}


# --- data (loaded once per process) -------------------------------------------------------
_CACHE = {}


def _rover_pca_arrays():
    """Pre-transform 40-D arrays cached by W02_Preprocessing_Pipeline, as the RF benchmark loads them."""
    if 'rover_pca' not in _CACHE:
        _CACHE['rover_pca'] = {
            'X_tr': np.load(TEMP_DIR / 'X_tr_raw.npy'), 'y_tr': np.load(TEMP_DIR / 'y_tr_s.npy'),
            'X_va': np.load(TEMP_DIR / 'X_va_raw.npy'), 'y_va': np.load(TEMP_DIR / 'y_va_s.npy'),
            'X_te': np.load(TEMP_DIR / 'X_te_raw.npy'), 'y_te': np.load(TEMP_DIR / 'y_te_s.npy'),
            'block_id_tr': np.load(TEMP_DIR / 'block_id_tr.npy'),
        }
    return _CACHE['rover_pca']


def _rover_tabular():
    """40-D raw+FFT+physical matrix rebuilt on the canonical split, standardised on train."""
    if 'rover_tab' not in _CACHE:
        df_clean = pd.read_csv(DATA_DIR / 'synthetic_rover_data.csv', index_col='timestamp',
                               parse_dates=True).ffill().bfill()
        feature_matrix_full, feature_names, row_idx_array, label_array_full = build_feature_matrix(df_clean)

        split_df = pd.read_csv(DATA_DIR / 'rover_stratified_block_split.csv').set_index('row_idx')
        purged = split_df.loc[row_idx_array, 'purged'].values
        split_array = split_df.loc[row_idx_array, 'split'].values[~purged]
        feature_matrix = feature_matrix_full[~purged]
        label_array = label_array_full[~purged]

        masks = {s: split_array == s for s in ('train', 'val', 'test')}
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feature_matrix[masks['train']]).astype(np.float32)
        _CACHE['rover_tab'] = {
            'X_tr': X_tr, 'y_tr': label_array[masks['train']],
            'X_va': scaler.transform(feature_matrix[masks['val']]).astype(np.float32),
            'y_va': label_array[masks['val']],
            'X_te': scaler.transform(feature_matrix[masks['test']]).astype(np.float32),
            'y_te': label_array[masks['test']],
            'scaler': scaler, 'feature_names': feature_names,
        }
    return _CACHE['rover_tab']


def _rover_windows():
    """w=50 window tensors, per-channel standardised with train-fold statistics.

    `seq` is (N, 50, 11) for the RNNs/Transformer; `conv` is the (N, 11, 50) transpose the CNN takes.
    """
    if 'rover_win' not in _CACHE:
        w = np.load(DATA_DIR / 'rover_windows.npz')
        Xr = {s: w[f'X_{s}_w{WINDOW}'] for s in ('train', 'val', 'test')}
        y = {s: w[f'y_{s}'].astype(np.int64) for s in ('train', 'val', 'test')}
        ch_mean = Xr['train'].reshape(-1, Xr['train'].shape[-1]).mean(0)
        ch_std = Xr['train'].reshape(-1, Xr['train'].shape[-1]).std(0) + 1e-8
        seq = {s: ((Xr[s] - ch_mean) / ch_std).astype(np.float32) for s in Xr}
        _CACHE['rover_win'] = {
            'seq': seq, 'conv': {s: np.transpose(seq[s], (0, 2, 1)).astype(np.float32) for s in seq},
            'y': y, 'scaler': {'channel_mean': ch_mean, 'channel_std': ch_std},
            'channels': [str(c) for c in w['sensor_col_names']],
        }
    return _CACHE['rover_win']


def _fari():
    """Fari interaction-quality task: stratified 70/15/15 by row, standardised on train."""
    if 'fari' not in _CACHE:
        feats = ['response_length', 'sentiment_score', 'topic_coherence', 'latency_ms', 'follow_up_rate']
        df = pd.read_csv(DATA_DIR / 'fari_interaction_quality.csv')
        Xf, yf = df[feats].values, df['good_interaction'].values
        X_tr_raw, X_tmp, y_tr, y_tmp = train_test_split(Xf, yf, test_size=0.30, stratify=yf,
                                                        random_state=SEED)
        X_va_raw, X_te_raw, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                          stratify=y_tmp, random_state=SEED)
        scaler = StandardScaler()
        _CACHE['fari'] = {
            'X_tr': scaler.fit_transform(X_tr_raw).astype(np.float32), 'y_tr': y_tr,
            'X_va': scaler.transform(X_va_raw).astype(np.float32), 'y_va': y_va,
            'X_te': scaler.transform(X_te_raw).astype(np.float32), 'y_te': y_te,
            'scaler': scaler, 'feature_names': feats,
        }
    return _CACHE['fari']


def _trajectory_data():
    """Humanoid motion tensors rebuilt from the persisted CSV (episode-level 70/15/15 split)."""
    if 'traj' not in _CACHE:
        df = pd.read_csv(DATA_DIR / 'synthetic_humanoid_motion.csv')
        n_seq = df['seq_id'].nunique()
        t_total = N_HIST + N_FUT

        def _grid(cols):
            return df[cols].values.reshape(n_seq, t_total, len(cols))

        P = _grid(['x_true', 'y_true'])
        P_meas = _grid(['x_meas', 'y_meas'])
        V_meas = _grid(['vx_meas', 'vy_meas'])
        heading_meas = _grid(['heading_meas'])[..., 0]
        d_sec = _grid(['d_left', 'd_center', 'd_right'])
        split = df['split'].values.reshape(n_seq, t_total)[:, 0]

        def build_xy(idx):
            Xp = P_meas[idx][:, :N_HIST].copy()
            Xp -= Xp[:, -1:, :]
            hd = np.unwrap(heading_meas[idx][:, :N_HIST], axis=1)
            X = np.concatenate([Xp, V_meas[idx][:, :N_HIST], hd[..., None],
                                d_sec[idx][:, :N_HIST]], axis=2)
            Y = P[idx][:, N_HIST:] - P_meas[idx][:, N_HIST - 1:N_HIST]
            return X.astype(np.float32), Y.astype(np.float32)

        idx_split = {s: np.where(split == s)[0] for s in ('train', 'val', 'test')}
        X_raw, Y = {}, {}
        for s, idx in idx_split.items():
            X_raw[s], Y[s] = build_xy(idx)
        f_mean = X_raw['train'].reshape(-1, 8).mean(0)
        f_std = X_raw['train'].reshape(-1, 8).std(0) + 1e-8
        _CACHE['traj'] = {
            'X': {s: ((X_raw[s] - f_mean) / f_std).astype(np.float32) for s in X_raw}, 'Y': Y,
            'idx_split': idx_split, 'V_meas': V_meas,
            'scaler': {'feature_mean': f_mean, 'feature_std': f_std},
            'feature_names': ['rel_x', 'rel_y', 'vx', 'vy', 'heading', 'd_left', 'd_center', 'd_right'],
        }
    return _CACHE['traj']


# --- Rover: Random Forest (W02_RF_Benchmark) ----------------------------------------------
def refit_rover_rf():
    """Pipeline(StandardScaler -> PCA(0.95) -> RF), grid-searched under StratifiedGroupKFold."""
    d = _rover_pca_arrays()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=SEED)),
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1)),
    ])
    param_grid = {'rf__n_estimators': [50, 100, 200], 'rf__max_depth': [None, 5, 10, 20]}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
    gs.fit(d['X_tr'], d['y_tr'], groups=d['block_id_tr'])

    model = gs.best_estimator_
    proba_va = model.predict_proba(d['X_va'])[:, 1]
    proba_te = model.predict_proba(d['X_te'])[:, 1]

    grid = np.linspace(0.1, 0.9, 81)   # the W02 benchmark's own threshold grid
    threshold = float(grid[int(np.argmax([f1_score(d['y_va'], proba_va >= t) for t in grid]))])

    config = {'seed': SEED, 'n_estimators': gs.best_params_['rf__n_estimators'],
              'max_depth': gs.best_params_['rf__max_depth'], 'class_weight': 'balanced',
              'pca_n_components': 0.95, 'pca_components_retained': int(model.named_steps['pca'].n_components_),
              'cv_best_f1': float(gs.best_score_), 'threshold_grid': [0.1, 0.9, 81]}
    return {'model': model, 'threshold': threshold, 'scaler': None,
            'metrics': _classification_metrics(d['y_te'], proba_te, threshold), 'config': config}


# --- Rover: MLP and 1D-CNN (W03_Neural_Network_Baseline) ----------------------------------
def _run_mlp(in_dim, hidden, activation, dropout, opt_name, X_tr, y_tr, X_va, y_va, seed=SEED):
    torch.manual_seed(seed)
    model = MLP(in_dim=in_dim, hidden=hidden, activation=activation, dropout=dropout)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(y_tr, DEVICE))
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    model, history, n_ep = train_with_early_stopping(model, make_loader(X_tr, y_tr), X_tr, y_tr,
                                                     X_va, y_va, criterion, optimizer, DEVICE)
    return model, history, n_ep


def refit_rover_mlp():
    """40 -> 64 -> 32 -> 2, ReLU, dropout 0.3, SGD(1e-2, momentum 0.9), class-weighted CE."""
    d = _rover_tabular()
    model, history, n_ep = _run_mlp(40, (64, 32), 'relu', 0.3, 'sgd_momentum',
                                    d['X_tr'], d['y_tr'], d['X_va'], d['y_va'])
    proba_va = predict_proba(model, d['X_va'], DEVICE)
    proba_te = predict_proba(model, d['X_te'], DEVICE)
    threshold, _ = tune_threshold(d['y_va'], proba_va)

    config = {'seed': SEED, 'in_dim': 40, 'hidden': [64, 32], 'activation': 'relu', 'dropout': 0.3,
              'optimizer': 'sgd_momentum', 'lr': 1e-2, 'momentum': 0.9, 'batch_size': 128,
              'max_epochs': 100, 'patience': 10, 'device': str(DEVICE),
              'epochs_run': n_ep, 'best_epoch': history['best_epoch'] + 1}
    return {'model': model.eval(), 'threshold': threshold, 'scaler': d['scaler'],
            'metrics': _classification_metrics(d['y_te'], proba_te, threshold), 'config': config}


def _run_cnn(X_tr, y_tr, X_va, y_va, pool, opt_name, seed=SEED):
    torch.manual_seed(seed)
    model = CNN1D(in_channels=X_tr.shape[1], pool=pool).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(y_tr, DEVICE))
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    model, history, n_ep = train_with_early_stopping(model, make_loader(X_tr, y_tr), X_tr, y_tr,
                                                     X_va, y_va, criterion, optimizer, DEVICE)
    return model, history, n_ep


def refit_rover_cnn():
    """2x Conv1d(k=3) -> concat avg+max pool -> FC head, SGD(1e-2, momentum 0.9)."""
    d = _rover_windows()
    model, history, n_ep = _run_cnn(d['conv']['train'], d['y']['train'], d['conv']['val'],
                                    d['y']['val'], pool='concat', opt_name='sgd_momentum')
    proba_va = predict_proba(model, d['conv']['val'], DEVICE)
    proba_te = predict_proba(model, d['conv']['test'], DEVICE)
    threshold, _ = tune_threshold(d['y']['val'], proba_va)

    config = {'seed': SEED, 'in_channels': 11, 'dropout': 0.3, 'pool': 'concat',
              'optimizer': 'sgd_momentum', 'lr': 1e-2, 'momentum': 0.9, 'batch_size': 128,
              'max_epochs': 100, 'patience': 10, 'window': WINDOW, 'device': str(DEVICE),
              'input_layout': '(batch, channel, time)',
              'epochs_run': n_ep, 'best_epoch': history['best_epoch'] + 1}
    return {'model': model.eval(), 'threshold': threshold, 'scaler': d['scaler'],
            'metrics': _classification_metrics(d['y']['test'], proba_te, threshold), 'config': config}


# --- Rover: LSTM / GRU / Transformer (W03_Sequence_Models) --------------------------------
def _run_rnn(rnn_type, X_tr, y_tr, X_va, y_va, seed=SEED):
    torch.manual_seed(seed)
    model = RNNClassifier(input_size=X_tr.shape[-1], hidden_size=64, rnn_type=rnn_type).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(y_tr, DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, history, n_ep = train_with_early_stopping(model, make_loader(X_tr, y_tr), X_tr, y_tr,
                                                     X_va, y_va, criterion, optimizer, DEVICE)
    return model, history, n_ep


def _refit_rover_rnn(rnn_type):
    d = _rover_windows()
    model, history, n_ep = _run_rnn(rnn_type, d['seq']['train'], d['y']['train'],
                                    d['seq']['val'], d['y']['val'])
    proba_va = predict_proba(model, d['seq']['val'], DEVICE)
    proba_te = predict_proba(model, d['seq']['test'], DEVICE)
    threshold, _ = tune_threshold(d['y']['val'], proba_va)

    config = {'seed': SEED, 'input_size': 11, 'hidden_size': 64, 'rnn_type': rnn_type,
              'dropout': 0.3, 'optimizer': 'adam', 'lr': 1e-3, 'batch_size': 128,
              'max_epochs': 100, 'patience': 10, 'window': WINDOW, 'device': str(DEVICE),
              'input_layout': '(batch, time, channel)',
              'epochs_run': n_ep, 'best_epoch': history['best_epoch'] + 1}
    return {'model': model.eval(), 'threshold': threshold, 'scaler': d['scaler'],
            'metrics': _classification_metrics(d['y']['test'], proba_te, threshold), 'config': config}


def refit_rover_lstm():
    """Single-layer unidirectional LSTM, hidden 64, final-state head, Adam(1e-3)."""
    return _refit_rover_rnn('lstm')


def refit_rover_gru():
    """Single-layer unidirectional GRU, hidden 64, final-state head, Adam(1e-3)."""
    return _refit_rover_rnn('gru')


def _run_transformer(X_tr, y_tr, X_va, y_va, seed=SEED, max_len=WINDOW):
    torch.manual_seed(seed)
    model = TransformerClassifier(input_size=X_tr.shape[-1], max_len=max_len).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(y_tr, DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model, history, n_ep = train_with_early_stopping(model, make_loader(X_tr, y_tr), X_tr, y_tr,
                                                     X_va, y_va, criterion, optimizer, DEVICE)
    return model, history, n_ep


def refit_rover_transformer():
    """input_proj 11->64, sinusoidal PE, 2-layer/4-head encoder, ff 128, mean-pool head."""
    d = _rover_windows()
    model, history, n_ep = _run_transformer(d['seq']['train'], d['y']['train'],
                                            d['seq']['val'], d['y']['val'])
    proba_va = predict_proba(model, d['seq']['val'], DEVICE)
    proba_te = predict_proba(model, d['seq']['test'], DEVICE)
    threshold, _ = tune_threshold(d['y']['val'], proba_va)

    config = {'seed': SEED, 'input_size': 11, 'd_model': 64, 'nhead': 4, 'num_layers': 2,
              'dim_feedforward': 128, 'dropout': 0.1, 'max_len': WINDOW, 'optimizer': 'adam',
              'lr': 1e-3, 'batch_size': 128, 'max_epochs': 100, 'patience': 10,
              'window': WINDOW, 'device': str(DEVICE), 'input_layout': '(batch, time, channel)',
              'epochs_run': n_ep, 'best_epoch': history['best_epoch'] + 1}
    return {'model': model.eval(), 'threshold': threshold, 'scaler': d['scaler'],
            'metrics': _classification_metrics(d['y']['test'], proba_te, threshold), 'config': config}


# --- Fari interaction quality (W03_Neural_Network_Baseline §4) -----------------------------
def refit_fari_rf():
    """RF grid-searched on the standardised 5-D Fari features (5-fold stratified CV)."""
    d = _fari()
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20]}
    gs = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1),
                      param_grid, cv=5, scoring='f1', n_jobs=-1)
    gs.fit(d['X_tr'], d['y_tr'])
    model = gs.best_estimator_

    proba_va = model.predict_proba(d['X_va'])[:, 1]
    proba_te = model.predict_proba(d['X_te'])[:, 1]
    threshold, _ = tune_threshold(d['y_va'], proba_va)

    config = {'seed': SEED, 'n_estimators': gs.best_params_['n_estimators'],
              'max_depth': gs.best_params_['max_depth'], 'class_weight': 'balanced',
              'cv_best_f1': float(gs.best_score_), 'split': 'stratified 70/15/15'}
    return {'model': model, 'threshold': threshold, 'scaler': d['scaler'],
            'metrics': _classification_metrics(d['y_te'], proba_te, threshold), 'config': config}


def refit_fari_mlp():
    """5 -> 16 -> 8 -> 2, ReLU, dropout 0.3, Adam(1e-3), class-weighted CE."""
    d = _fari()
    model, history, n_ep = _run_mlp(5, (16, 8), 'relu', 0.3, 'adam',
                                    d['X_tr'], d['y_tr'], d['X_va'], d['y_va'])
    proba_va = predict_proba(model, d['X_va'], DEVICE)
    proba_te = predict_proba(model, d['X_te'], DEVICE)
    threshold, _ = tune_threshold(d['y_va'], proba_va)

    config = {'seed': SEED, 'in_dim': 5, 'hidden': [16, 8], 'activation': 'relu', 'dropout': 0.3,
              'optimizer': 'adam', 'lr': 1e-3, 'batch_size': 128, 'max_epochs': 100, 'patience': 10,
              'device': str(DEVICE), 'split': 'stratified 70/15/15',
              'epochs_run': n_ep, 'best_epoch': history['best_epoch'] + 1}
    return {'model': model.eval(), 'threshold': threshold, 'scaler': d['scaler'],
            'metrics': _classification_metrics(d['y_te'], proba_te, threshold), 'config': config}


# --- Trajectory regressors (W04_Trajectory_Prediction) -------------------------------------
def _train_regressor(model, X_tr, Y_tr, X_va, Y_va, max_epochs=100, patience=10, weight_decay=0.0):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)),
                        batch_size=128, shuffle=True)
    Xtr_t, Ytr_t = torch.tensor(X_tr, device=DEVICE), torch.tensor(Y_tr, device=DEVICE)
    Xva_t, Yva_t = torch.tensor(X_va, device=DEVICE), torch.tensor(Y_va, device=DEVICE)
    best_val, best_state, best_epoch, n_bad = np.inf, None, -1, 0
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            history['train_loss'].append(criterion(model(Xtr_t), Ytr_t).item())
            va_loss = criterion(model(Xva_t), Yva_t).item()
        history['val_loss'].append(va_loss)
        if va_loss < best_val - 1e-7:
            best_val, best_epoch, n_bad = va_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            n_bad += 1
            if n_bad >= patience:
                break
    model.load_state_dict(best_state)
    history['best_epoch'] = best_epoch
    return model, history


def _euclid_per_horizon(Y_true, Y_pred):
    d = np.linalg.norm(Y_pred - Y_true, axis=2)
    return d.mean(axis=0), d.mean()


def _predict_torch(model, Xs):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xs, device=DEVICE)).cpu().numpy()


def _traj_metrics(Y_true, Y_pred):
    hor, overall = _euclid_per_horizon(Y_true, Y_pred)
    return {'mean_cm': float(overall * 100), 'per_horizon_cm': [float(v * 100) for v in hor]}


def refit_trajectory():
    """CV floor, closed-form Linear, MLP, LSTM and Transformer seq2seq at the canonical seed 42."""
    d = _trajectory_data()
    X, Y = d['X'], d['Y']

    models, metrics, configs = {}, {}, {}

    # constant-velocity extrapolation: zero-parameter physics floor
    v_last = d['V_meas'][d['idx_split']['test']][:, N_HIST - 1]
    steps = (np.arange(1, N_FUT + 1) * DT)[None, :, None]
    models['cv'] = None
    metrics['cv'] = _traj_metrics(Y['test'], v_last[:, None, :] * steps)
    configs['cv'] = {'n_params': 0, 'dt': DT, 'n_hist': N_HIST, 'n_fut': N_FUT,
                     'rule': 'last measured velocity carried forward from the last measured pose'}

    linreg = LinearRegression().fit(X['train'].reshape(len(X['train']), -1),
                                    Y['train'].reshape(len(Y['train']), -1))
    models['linear'] = linreg
    metrics['linear'] = _traj_metrics(
        Y['test'], linreg.predict(X['test'].reshape(len(X['test']), -1)).reshape(-1, N_FUT, 2))
    configs['linear'] = {'n_params': int(linreg.coef_.size + linreg.intercept_.size),
                         'input': 'flattened standardised 10x8 history', 'fit': 'closed form'}

    factory = {'mlp': MLPRegressor, 'lstm': LSTMRegressor, 'transformer': TrajTransformer}
    for name, cls in factory.items():
        torch.manual_seed(SEED)
        model, history = _train_regressor(cls(), X['train'], Y['train'], X['val'], Y['val'])
        models[name] = model.eval()
        metrics[name] = _traj_metrics(Y['test'], _predict_torch(model, X['test']))
        configs[name] = {'seed': SEED, 'optimizer': 'adam', 'lr': 1e-3, 'batch_size': 128,
                         'max_epochs': 100, 'patience': 10, 'loss': 'mse', 'device': str(DEVICE),
                         'n_params': int(sum(p.numel() for p in model.parameters())),
                         'epochs_run': len(history['train_loss']),
                         'best_epoch': history['best_epoch'] + 1}
    configs['mlp'].update({'n_hist': N_HIST, 'n_feat': 8, 'n_fut': N_FUT, 'dropout': 0.1})
    configs['lstm'].update({'n_feat': 8, 'hidden': 64, 'n_fut': N_FUT})
    configs['transformer'].update({'n_feat': 8, 'd_model': 64, 'nhead': 4, 'num_layers': 2,
                                   'dim_feedforward': 128, 'dropout': 0.1, 'n_hist': N_HIST,
                                   'n_fut': N_FUT})

    return {'models': models, 'metrics': metrics, 'scalers': d['scaler'], 'config': configs}


REFITS = {
    ('rover', 'rf'): refit_rover_rf,
    ('rover', 'mlp'): refit_rover_mlp,
    ('rover', 'cnn'): refit_rover_cnn,
    ('rover', 'lstm'): refit_rover_lstm,
    ('rover', 'gru'): refit_rover_gru,
    ('rover', 'transformer'): refit_rover_transformer,
    ('fari', 'rf'): refit_fari_rf,
    ('fari', 'mlp'): refit_fari_mlp,
}


# --- persistence ---------------------------------------------------------------------------
def _model_path(family, name):
    ext = '.joblib' if (family, name) not in TORCH_CLASSES else '.pt'
    return SAVED_DIR / family / f'{name}{ext}'


def _scaler_path(family, name):
    return SAVED_DIR / family / f'{name}_scaler.joblib'


def _save_one(family, name, model, scaler, config):
    path = _model_path(family, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.pt':
        torch.save({'state_dict': model.state_dict(), 'config': config}, path)
    else:
        joblib.dump(model, path)
    if scaler is not None:
        joblib.dump(scaler, _scaler_path(family, name))
    return path


def _build_torch(family, name, config):
    cls = TORCH_CLASSES[(family, name)]
    kwargs = {k: config[k] for k in _INIT_KEYS[cls] if k in config}
    if 'hidden' in kwargs and isinstance(kwargs['hidden'], list):
        kwargs['hidden'] = tuple(kwargs['hidden'])
    return cls(**kwargs)


def load_model(family, name):
    """Reload a saved model with its threshold, scaler, canonical metrics and config.

    Returns the same dict shape the refit functions return, one model at a time — for the
    trajectory family the metrics/config are that model's entry rather than the full set.
    ``load_model('trajectory', 'cv')`` returns ``model=None``: the constant-velocity floor is a
    rule with no parameters, so only its config and metrics are stored.
    """
    entry = json.loads(MANIFEST.read_text())[f'{family}/{name}']
    scaler_path = _scaler_path(family, name)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    if entry['file'] is None:
        model = None
    else:
        path = SAVED_DIR / entry['file']
        if path.suffix == '.pt':
            blob = torch.load(path, weights_only=False, map_location=DEVICE)
            model = _build_torch(family, name, blob['config'])
            model.load_state_dict(blob['state_dict'])
            model.to(DEVICE).eval()
        else:
            model = joblib.load(path)
    return {'model': model, 'threshold': entry['threshold'], 'scaler': scaler,
            'metrics': entry['metrics'], 'config': entry['config']}


def _test_predictions(family, name, model):
    """Test-set output used for the load/refit round-trip check."""
    if family == 'rover' and name == 'rf':
        return model.predict_proba(_rover_pca_arrays()['X_te'])[:, 1]
    if family == 'rover' and name == 'mlp':
        return predict_proba(model, _rover_tabular()['X_te'], DEVICE)
    if family == 'rover' and name == 'cnn':
        return predict_proba(model, _rover_windows()['conv']['test'], DEVICE)
    if family == 'rover':
        return predict_proba(model, _rover_windows()['seq']['test'], DEVICE)
    if family == 'fari' and name == 'rf':
        return model.predict_proba(_fari()['X_te'])[:, 1]
    if family == 'fari':
        return predict_proba(model, _fari()['X_te'], DEVICE)
    d = _trajectory_data()
    if name == 'cv':
        v_last = d['V_meas'][d['idx_split']['test']][:, N_HIST - 1]
        return v_last[:, None, :] * (np.arange(1, N_FUT + 1) * DT)[None, :, None]
    if name == 'linear':
        Xt = d['X']['test']
        return model.predict(Xt.reshape(len(Xt), -1)).reshape(-1, N_FUT, 2)
    return _predict_torch(model, d['X']['test'])


def _verify(family, name, metrics, threshold):
    """Compare against the published notebook values; returns (target, achieved, |delta|) rows."""
    rows = []
    for key, target in TARGETS[(family, name)].items():
        achieved = threshold if key == 'threshold' else metrics[key]
        rows.append({'metric': key, 'target': target, 'achieved': float(achieved),
                     'abs_delta': abs(float(achieved) - target)})
    return rows


def save_all():
    """Run every refit, verify against the published values, save checkpoints and the manifest."""
    SAVED_DIR.mkdir(exist_ok=True)
    manifest, fresh_predictions = {}, {}

    def _record(family, name, model, threshold, scaler, metrics, config):
        key = f'{family}/{name}'
        path = _save_one(family, name, model, scaler, config) if model is not None else None
        manifest[key] = {
            'file': str(path.relative_to(SAVED_DIR)) if path else None,
            'file_bytes': path.stat().st_size if path else 0,
            'scaler_file': (str(_scaler_path(family, name).relative_to(SAVED_DIR))
                            if scaler is not None else None),
            'threshold': threshold, 'config': config, 'metrics': metrics,
            'verification': _verify(family, name, metrics, threshold),
        }
        fresh_predictions[key] = (_test_predictions(family, name, model)
                                  if model is not None else None)

    for (family, name), fn in REFITS.items():
        r = fn()
        _record(family, name, r['model'], r['threshold'], r['scaler'], r['metrics'], r['config'])

    traj = refit_trajectory()
    for name, model in traj['models'].items():
        _record('trajectory', name, model, None, traj['scalers'] if model is not None else None,
                traj['metrics'][name], traj['config'][name])

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # round trip: the reloaded object must reproduce the fresh fit's test predictions exactly
    for key, pred in fresh_predictions.items():
        family, name = key.split('/')
        reloaded = load_model(family, name)
        if reloaded['model'] is None:
            manifest[key]['round_trip_max_abs_diff'] = None
            continue
        again = _test_predictions(family, name, reloaded['model'])
        manifest[key]['round_trip_max_abs_diff'] = float(np.max(np.abs(again - pred)))
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    _print_report(manifest)
    return manifest


def _print_report(manifest):
    print(f'\ndevice: {DEVICE}   saved under: {SAVED_DIR}')
    header = f'{"model":24s} {"metric":12s} {"target":>10s} {"achieved":>10s} {"|delta|":>9s}  {"round-trip":>10s}'
    print(header)
    print('-' * len(header))
    for key in sorted(manifest):
        entry = manifest[key]
        diff = entry.get('round_trip_max_abs_diff')
        for i, c in enumerate(entry['verification']):
            rt = ('n/a' if diff is None else f'{diff:.2e}') if i == 0 else ''
            print(f'{key if i == 0 else "":24s} {c["metric"]:12s} {c["target"]:10.4f} '
                  f'{c["achieved"]:10.4f} {c["abs_delta"]:9.4f}  {rt:>10s}')


if __name__ == '__main__':
    save_all()
