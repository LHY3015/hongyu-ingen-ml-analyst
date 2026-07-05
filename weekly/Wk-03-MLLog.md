# Week 03 ML Log — Neural Network Baseline

*Covers the MLP + 1D-CNN + Fari generalisation task. Sequence/Transformer models to be appended.*

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
