# ML & Neural Network Analyst Intern @InGen Dynamics Inc.




## Program Overview

This repository contains internship deliverables for task submission, self-learning, and knowledge sharing. It covers model development, training, and evaluation on data from the following platforms:

| Platform | Domain |
|---|---|
| Aido Rover | Outdoor patrol (primary data anchor) |
| Sentinel Prime AI | Security / anomaly detection |
| Fari | Eldercare companion |
| Aido Humanoid | Bipedal research / trajectory prediction |
| Senpai | Educational robot |





## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/          # Weekly notebooks (W01–W08), named WXX_TopicSnakeCase.ipynb
├── data/               # Synthetic sensor datasets (no proprietary data)
├── reports/            # Markdown analysis reports (W02, W03, W06, W07)
├── capstone/           # W08 capstone report + deck + retrospective
└── weekly/             # Weekly ML logs (Wk-NN-MLLog.md)
```


## Notebooks

| Week | Notebook |  summary |
|---|---|---|
| 1 | `W01_env_check.ipynb` | Verify Python/sklearn/PyTorch/pandas/scipy toolchain |
| 2 | `W02_Preprocessing_Pipeline.ipynb` | Synthetic Aido Rover sensor data: FFT, PCA, RF feature selection |
| 3 | `W03_Classical_ML_Benchmark.ipynb` | 5-model comparison (DT, SVM, RF, Bayesian, KMeans) with full metric table |
| 4 | `W04_Neural_Network_Baseline.ipynb` | MLP + 1D-CNN training, learning curves, latency comparison |
| 5 | `W05_LSTM_TimeSeries.ipynb` | LSTM/GRU anomaly detection; latency vs. window size |
| 5 | `W05_Trajectory_Pred_Notebook.ipynb` | ML-assisted trajectory prediction (linear / MLP / LSTM regressors) |
| 6 | `W06_Model_Explainability.ipynb` | SHAP feature importance + learning curves across 3 models × 3 tasks |
| 7 | `W07_Optimisation_Notebook.ipynb` | Latency-accuracy Pareto frontier; model pruning / quantisation experiment |

---

## Reports

| File | Description |
|---|---|
| `reports/W01_PhysicalAI_ML_Landscape.md` | 5-page brief: InGen platforms × ML analyst lens; PIC 2.0 ML module map; CNC-to-robot pipeline bridge |
| `reports/W02_Feature_Analysis_Report.md` | 2-page feature analysis report: selected features, PCA variance explained, RF importance ranking |
| `reports/W03_Model_Comparison_Report.md` | 3-page model selection rationale; confusion matrices; latency table |
| `reports/W06_PIC20_ML_Analysis.md` | 4-page analysis mapping all 6 PIC 2.0 model classes to experimental findings |
| `reports/W07_Methodology_Report.md` | 4-page publication-standard pipeline + evaluation + reproducibility report |
| `capstone/W08_Capstone_Report.docx` | 15–18 page technical ML analysis report |
| `capstone/W08_Capstone_Deck.pptx` | 10-slide executive deck with finding-statement headlines |
| `capstone/W08_Retrospective.md` | 1-page retrospective + thesis connection note |


## Toolchain

| Category | Tools |
|---|---|
| Core Python & Data | Python 3.11, pandas, NumPy, SciPy |
| Classical ML | scikit-learn (DT, SVM, RF, GaussianNB, KMeans, PCA) |
| Neural Networks | PyTorch (MLP, Conv1d, LSTM, GRU) |
| Explainability | SHAP (TreeExplainer, DeepExplainer, KernelExplainer) |
| Evaluation | scikit-learn metrics, timeit, pytest |
| Visualisation | matplotlib, seaborn |

## Quick Start

### 1.Initialize repo
```bash
mkdir -p ~/MY_WORKSPACE && cd ~/MY_WORKSPACE
git clone https://github.com/LHY3015/hongyu-ingen-ml-analyst.git
cd hongyu-ingen-ml-analyst
```
### 2.Install dependencies:
```bash
pip install -r requirements.txt
```
### 3.Install PyTorch
Install the build that matches your hardware (see https://pytorch.org/get-started/locally/):
```bash
# CUDA (replace cuXXX with your version, e.g. cu130 for CUDA 13.0):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
# CPU only (already installed by step 2, no extra action needed)
```
### 4.Run
Open any `notebooks/WXX_*.ipynb` and run all cells. Seed values are documented at the top of each notebook.
