# Week 03 ML Log — Neural Network Baseline + Sequence Models

*First half: MLP + 1D-CNN + Fari generalisation task. Second half (appended below): LSTM / GRU / Transformer.*

## What was done

- MLP: `40→64→32→2` on the 40-D raw+FFT+physical matrix; activation/dropout/optimiser each selected via 3-seed ablation on mean val loss — ReLU, dropout 0.3, SGD+momentum(0.9); all three deltas sit within 1 std (near-ties).

- 1D-CNN: `Conv1d(11→32,k=3)→Conv1d(32→64,k=3)`, `k=3` sized to the onset transient (0.5 s receptive field vs. 3.1 s mean anomaly duration), pooling integrates over the full 5 s window.
- Pooling head (avg/max/concat): canonical-fold 3-seed screen promoted concat + max to the 7-fold rotation; **concat wins** (7-fold F1 0.789±0.059 vs. max's 0.677±0.166 — max looked fine on one fold but is unstable across folds).
- 1D-CNN optimiser (Adam vs SGD+momentum) also ablated (3-seed, canonical fold): SGD+momentum wins on mean val loss and replaces Adam for the final CNN — but the 7-fold rotation then shows this trades stability for the lower val loss (F1 0.731±0.115 vs. the Adam config's 0.789±0.059, nearly 2x the std), the same canonical-fold-underdetects-instability pattern the pooling check above was designed to catch.
- Both models use class-weighted CE + val-tuned decision threshold (max F1 on val, applied to test) — same protocol as the Week-2 RF benchmark.
- Tried: Platt-scaling calibration on val probabilities before threshold tuning (RF/MLP/CNN, uniformly) to address the CNN's fold-3 instability above. Result: no meaningful change for RF (0.7492±0.0481 → 0.7505±0.0537) or MLP (0.7936±0.0472 → 0.7930±0.0549), and it made the CNN worse (0.7307±0.1147 → 0.7119±0.1787, fold 3 tuned-F1 dropped further to 0.324) — the fold-3 failure is a val/test distribution mismatch for that specific rotation, not a systematic calibration bias, so refitting on the same val fold only adds overfitting freedom rather than fixing it. Not adopted; ledger/report keep the pre-calibration numbers above.
- Fari second task: 3,000 synthetic samples, 5 non-sensor features, label = noisy logistic draw (Bayes ceiling F1=0.7867); plain stratified 70/15/15 split (no temporal leakage risk here). RF (grid search) and MLP run unmodified on this task to test generalisation.
- Model ledger (`data/model_ledger.csv`) started: RF (carried forward), MLP, 1D-CNN, each reporting 7-fold mean±std F1/AUC at a fixed, already-selected config, per the project's two-phase fold-evaluation protocol. Fari not appended — no plan-defined latency gate for it.

## Results

| Model                         | 7-fold F1 (mean±std) | 7-fold AUC (mean±std) | Latency (single, CPU) | Verdict (≤100 ms) |
| ----------------------------- | --------------------- | ---------------------- | --------------------- | ------------------ |
| RandomForest                  | 0.7492 ± 0.0481      | 0.9595 ± 0.0092       | 7.856 ms              | PASS               |
| MLP                           | 0.7936 ± 0.0472      | 0.9786 ± 0.0112       | 0.135 ms              | PASS               |
| 1D-CNN (concat, SGD+momentum) | 0.7307 ± 0.1147      | 0.9648 ± 0.0221       | 0.152 ms              | PASS               |

| Task | Model | Test F1 (tuned) | Test AUC |
| ---- | ----- | --------------- | -------- |
| Fari | RF    | 0.7559          | 0.8077   |
| Fari | MLP   | 0.7555          | 0.8229   |

Both within ~0.03-0.04 F1 of the 0.7867 Bayes ceiling.

## Deliverables Completed

- `W03_Neural_Network_Baseline.ipynb` — 40-D matrix, MLP ablations + 7-fold rotation, 1D-CNN pooling selectio, 7-fold rotation, Fari task, all confusion matrices/reports/latency
- `W03_Neural_Network_Baseline.md` — design rationale, evaluation, cross-model comparison
- `fari_interaction_quality.csv`, `model_ledger.csv`

---

## Second Half — Sequence Models (LSTM / GRU / Transformer)

## What was done

- LSTM, GRU (unidirectional, hidden 64) and a 2-layer/4-head Transformer encoder (d_model 64, sinusoidal PE, mean-pool head) trained on the Week-2 w=50 tensor under the identical first-half protocol; window size justified quantitatively (w=50 fully contains 87.5% of fault runs vs 2.8% at w=10); all three retrained across the 7-fold block rotation for the ledger.
- Head-to-head: convergence, 7-fold F1/AUC, data efficiency (25/50/100% × 3 seeds), latency-vs-window sweep with scaling-law crossover extrapolation.
- Attention interpretability anchored to ground truth: per-step fault masks reconstructed from the canonical split; attention/hidden-state selectivity quantified over all anomalous test windows; misses stratified by in-window fault coverage and fault-run length (fault-type proxy).
- `W03_Sequence_Comparison_Report.md` written; ledger extended (LSTM/GRU/Transformer rows with per-fold columns).

## Results

| Model | Params | 7-fold F1 (mean±std) | 7-fold AUC (mean±std) | Latency (single, CPU) | Verdict (≤100 ms) |
| --- | --- | --- | --- | --- | --- |
| LSTM | 19.8K | 0.7038 ± 0.1590 | 0.9694 ± 0.0137 | 0.191 ms | PASS |
| GRU | 14.9K | 0.7468 ± 0.0818 | 0.9747 ± 0.0062 | 0.666 ms | PASS |
| Transformer (d=64) | 67.8K | 0.8289 ± 0.0688 | 0.9822 ± 0.0045 | 0.519 ms | PASS |
| Transformer-S (d=32, control) | 17.5K | 0.7604 ± 0.0522 | 0.9698 ± 0.0138 | 0.312 ms | PASS |

- Transformer has the best single-number F1/AUC, but **paired per-fold t-tests + a param-matched control show detection quality does not separate the architectures**: the Transformer's edge is not significant vs MLP (p=0.34, MLP's worst fold is better), only borderline vs GRU (p=0.05), and at matched params (Transformer-S, d_model=32) F1 drops 0.829→0.760 and ties the GRU (p=0.76). MLP/GRU/RF/Transformer-S are one near-tie; only the 67.8K-param Transformer pokes above, at 4.5× the weights. Revised deploy rec: **MLP is the default**; Transformer is an interpretability-motivated alternate. RNNs converge in 2–5 epochs vs 18 for the Transformer; GRU beats LSTM decisively on fold stability (LSTM ±0.159 = worst in ledger, a threshold-calibration failure, not a ranking one).
- **Lesson (methodology):** the head-to-head first matched *width* (hidden/d_model = 64) and the report initially read that as "compares architectures, not parameters" — wrong, since a 2-layer attention encoder holds 4.5× the weights at equal width. Fixed by adding a param-matched control (Transformer-S) and paired significance tests; both reports/deck now frame the Transformer's lead as capacity, not mechanism. General rule for the ledger: equal width ≠ equal capacity; name a winner only with a significance test.
- **How attention reweights the sequence vs the LSTM hidden state:** attention mass lands directly on the true fault span (median fault/normal selectivity 5.3×, >1 in 98% of *eligible* anomalous windows — with no per-step labels), while the LSTM registers the fault only as a hidden-state regime shift whose per-step change barely favours fault steps (1.15×, 57%) — a causal state smears evidence forward, attention pins it in place. This is a genuine **localisation/interpretability** advantage (points at *where* the fault is); it does not, per the controls above, translate into better per-parameter detection.
- Miss analysis: residual FNs are dominated by steady-state stuck-type faults (39.5% caught vs 76–93% for slip-range/ambiguous) — all four torques rise uniformly, no cross-wheel asymmetry, indistinguishable from heavy terrain once the onset leaves the buffer; plus a ~2% irreducible floor (fault starting exactly at the anchor). Recall ceiling is a feature/context problem, queued for Phase D.
- Latency never binds: extrapolated 100 ms crossovers sit at hundreds of seconds of buffer; window choice is a pure detection-quality decision at this platform's scale.

## Deliverables Completed

- `W03_Sequence_Models_RNN_vs_Transformer.ipynb` — LSTM/GRU/Transformer, 7-fold rotation, data efficiency (3-seed bands), latency-vs-window, ground-truth-anchored attention analysis
- `W03_Sequence_Comparison_Report.md`
- `model_ledger.csv` — six models, per-fold F1/AUC columns, all PASS
