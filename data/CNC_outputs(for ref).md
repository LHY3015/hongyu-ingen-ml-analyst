# CNC Wear Detection — Model Outputs

Source: `local/CNC_project.ipynb`  
Dataset: 18 experiments (exp_01–18); train = exp 01–15, test = exp 16–18  
Task: Binary classification — tool worn (1) vs. unworn (0)  
Test set size: 2,692 samples (1,177 class-0, 1,515 class-1); real-time freq domain test: 268 samples

---

## 1. Baseline RF — Feature Space Comparison (cell 25)

Random Forest (default params, random_state=42), evaluated on test set.

| Feature Space          | Test Accuracy |
|------------------------|--------------|
| Time Domain (Original) | **98.48%**   |
| Time Domain (PCA, 5)   | 49.55%       |
| Time Domain (LDA, 1)   | 47.99%       |
| Time Domain (LLE, 3)   | 49.67%       |
| Frequency Domain       | 75.22%       |

**Time Domain (Original) Classification Report:**
```
               precision    recall  f1-score   support
           0       0.97      0.99      0.98      1177
           1       0.99      0.98      0.99      1515
    accuracy                           0.98      2692
```

> PCA/LDA/LLE on time-domain features collapse to ~50% — dimensionality reduction hurts here because the signal lies in feature magnitude, not variance direction.

---

## 2. RF Feature Selection — Top-K Raw Features (cell 27)

Feature importance ranked by RF on full time-domain data. Top-2 features: **Z1_CommandPosition**, **Z1_ActualPosition**.

| Reduction | Features Used | Test Accuracy | Notes |
|-----------|--------------|--------------|-------|
| 1         | Z1_CommandPosition | 97.51% | single feature |
| 2         | Z1_CommandPosition + Z1_ActualPosition | **97.81%** | minimal practical set |
| 10        | top-10 features | 97.81% | no gain over top-2 |
| 20        | top-20 features | 98.70% | |
| 30        | top-30 features | **99.03%** | peak accuracy |
| 40        | top-40 features | 98.51% | slight overfit |

**Reduction 2 Classification Report:**
```
               precision    recall  f1-score   support
           0       0.98      0.97      0.97      1177
           1       0.98      0.98      0.98      1515
    accuracy                           0.98      2692
```

---

## 3. Real-time Frequency Domain RF (cell 31)

FFT applied per 10-sample window; top-3 freq+amp features per channel.

| Metric | Value |
|--------|-------|
| Test Accuracy | 63.43% |
| Class-0 F1 | 0.53 |
| Class-1 F1 | 0.70 |

> Much lower than time-domain — windowed FFT on 10 samples at 10 Hz is too short for stable spectral estimation.

---

## 4. Multi-Model Comparison (cell 33)

Grid search + 5-fold CV + SMOTE. Scoring = accuracy (Clustering = silhouette score).

### Time Domain (Original)
| Model | Best Params | Test Accuracy |
|-------|-------------|--------------|
| Clustering | — | silhouette 0.6352 |
| Decision Tree | criterion=entropy, max_depth=20, min_samples_split=10 | **97.55%** |
| SVM | C=50, kernel=rbf, gamma=scale | 59.84% |
| ANN | hidden=(200,), activation=relu, alpha=0.01 | 50.89% |
| Bayesian | — | 46.36% |

### Frequency Domain (global FFT)
| Model | Best Params | Test Accuracy |
|-------|-------------|--------------|
| Clustering | — | silhouette 0.8301 |
| Decision Tree | criterion=entropy, max_depth=10, min_samples_split=2 | **91.89%** |
| SVM | C=50, kernel=rbf, gamma=scale | 73.59% |
| ANN | hidden=(100,), activation=relu, alpha=0.001 | 75.00% |
| Bayesian | — | 44.49% |

### Frequency Domain (real-time, 10-sample windows)
| Model | Best Params | Test Accuracy |
|-------|-------------|--------------|
| Clustering | — | silhouette 0.5782 |
| Decision Tree | criterion=gini, max_depth=10, min_samples_split=10 | 66.79% |
| SVM | C=50, kernel=rbf, gamma=scale | 55.97% |
| ANN | hidden=(200,), activation=relu, alpha=0.0001 | 57.46% |
| Bayesian | — | 47.76% |

### Reduction 1 (Z1_CommandPosition only)
| Model | Test Accuracy |
|-------|--------------|
| Clustering | silhouette 0.6654 |
| Decision Tree | **98.11%** |
| SVM | 81.76% |
| ANN | 66.72% |
| Bayesian | 70.99% |

### Reduction 2 (Z1_CommandPosition + Z1_ActualPosition)
| Model | Test Accuracy |
|-------|--------------|
| Clustering | silhouette 0.6652 |
| Decision Tree | **98.03%** |
| SVM | 81.76% |
| ANN | **98.03%** |
| Bayesian | 70.99% |

### Reduction 10 (top-10 features)
| Model | Test Accuracy |
|-------|--------------|
| Clustering | silhouette 0.6447 |
| Decision Tree | **98.14%** |
| SVM | 77.86% |
| ANN | 87.82% |
| Bayesian | 46.29% |

---

## Key Takeaways

- **Best overall**: RF / Decision Tree on time-domain original features (~98–99%)
- **Minimal viable set**: 2 features (Z1 position) → 97.81% test acc; Decision Tree on same → 98.03%
- **SVM/ANN struggle** on full time-domain (likely scale-sensitive to Z1 position magnitude)
- **Frequency domain** underperforms time domain for this task; real-time FFT (10-sample) is particularly weak
- **RF is the reference model** for CNC wear detection in this project
