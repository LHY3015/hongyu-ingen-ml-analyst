# Machine Learning & Neural Network Analyst Intern  @InGen Dynamics Inc.

## Program Overview

This repository contains internship deliverables for task submission, self-learning, and knowledge sharing. It covers model development, training and evaluation for the following platforms, under Origami AI Physical Intelligence Platform (PIC 2.0):

## Program Arc

| Phase       | Weeks | Theme                                                                         |
| ----------- | ----- | ----------------------------------------------------------------------------- |
| **A** | 1–2  | Landscape, Data Preprocessing, Classic ml benchmark,Sequence/RL Scaffolding |
| **B** | 3–4  | NN Baseline, Sequence Models, Transformers, Trajectory Prediction            |
| **C** | 5–6  | Reinforcement Learning                                                        |
| **D** | 7–8  | Explainability, Pareto Optimisation, Methodology, Capstone                   |

## Platforms

| Platform          | Domain                                                        |
| ----------------- | ------------------------------------------------------------- |
| Origami / PIC 2.0 | All 6 model classes: GRPO, STUM, SEOM, AMDC, HTD-IRL, CRL-MRS |
| Aido Rover        | Outdoor patrol (primary data anchor)                          |
| Sentinel Prime AI | Security / anomaly detection                                  |
| Fari              | Eldercare companion                                           |
| Aido Humanoid     | Bipedal research / trajectory prediction                      |
| Senpai            | Educational robot                                             |

## Repository Structure

```
.
├── README.md  
├── requirements.txt	# Dependencies
├── notebooks/          # Weekly notebooks (W01–W08), named WXX_<TopicSnakeCase>.ipynb
├── data/               # Synthetic sensor datasets and RL transition tables
├── reports/            # Analysis reports
├── rl/                 # RL environments, policies and MDP schema
├── capstone/           # W08 capstone report + deck + retrospective
└── weekly/             # Weekly logs, named Wk-NN-MLLog.md
```

## Notebooks

| Week | Notebook                                         | Summary                                                                                                                                       |
| ---- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `W01_env_check.ipynb`                          | Verify Python / scikit-learn / PyTorch / pandas / SciPy / Gymnasium toolchain                                                                 |
| 2    | `W02_Preprocessing_Pipeline.ipynb`             | FFT, PCA, RF feature selection; synthetic Aido Rover dataset generation                                                                       |
| 2    | `W02_RF_Benchmark.ipynb`                       | RF anomaly classification: metric table, confusion matrix, latency, ROC; imbalance handling                                                  |
| 2    | `W02_Sequence_and_RL_Scaffolding.ipynb`        | Sliding-window tensor builder (rover_windows.npz) + MDP transition-table generator (rover_transitions.csv)                                    |
| 3    | `W03_Neural_Network_Baseline.ipynb`            | MLP + CNN training, learning curves, latency                                                                                                  |
| 3    | `W03_Sequence_Models_RNN_vs_Transformer.ipynb` | LSTM vs GRU vs Transformer; attention-map interpretability; latency vs window size                                                           |
| 4    | `W04_Trajectory_Prediction.ipynb`              | Trajectory prediction: Linear / MLP / LSTM / Transformer seq2seq;<br />per-horizon error; attention over horizon; latency vs 50 ms constraint |
| 5    | `W05_RL_Environment.ipynb`                     | Gymnasium env build from Week-2 MDP schema; random-rollout validation; reward sanity checks                                                   |
| 5    | `W05_Value_Based_RL.ipynb`                     | Q-learning + DQN; learning curves with seed-variance bands; return mean ± std (≥5 seeds); reward-shaping ablation                           |
| 5    | `W05_Offline_RL_BC.ipynb`                      | Behaviour cloning + offline value estimate from Week-2 transition table; comparison to online DQN                                             |
| 6    | `W06_PolicyGradient_RL.ipynb`                  | REINFORCE, PPO, GRPO-style group-relative baseline; sample-efficiency comparison; policy inference latency                                    |
| 6    | `W06_IRL_Hierarchical.ipynb`                   | Reward recovery from expert trajectories; hierarchical-RL discussion (HTD-IRL connection)                                                     |
| 6    | `W06_MultiAgent_RL.ipynb`                      | PettingZoo two-agent cooperative patrol; coordination and credit-assignment analysis (CRL-MRS)                                                |
| 7    | `W07_Explainability.ipynb`                     | SHAP (classical + DL); attention-map recap; RL value/saliency; Senpai 3-class learner-state task                                              |
| 7    | `W07_Pareto_and_Optimisation.ipynb`            | Full cross-family latency-accuracy Pareto frontier; one optimisation experiment (quantisation / distillation) with before/after latency       |

## Reports

| Week | File                                  | Description                                                                                                                                                        |
| ---- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | `W01_PhysicalAI_ML_Landscape.md`    | 5-page brief: per-platform ML brief; PIC 2.0 ML class map; 6-paper literature summary                                                                              |
| 2    | `W02_Feature_Analysis_Report.md`    | 2-page feature analysis: data quality, PCA variance/loadings, RF importance and selected feature subset, CNC comparison                                            |
| 3    | `W03_Sequence_Comparison_Report.md` | 3-page RNN vs Transformer Encoder: anomaly-detection accuracy, latency analysis, attention interpretability                                                        |
| 4    | `W04_Trajectory_Report.md`          | 2–3 page: trajectory model comparison, per-horizon error analysis, Transformer vs LSTM tradeoff, quadrotor thesis connection                                      |
| 4    | `W04_Mid_Review_Deck.pptx`          | 6–8 slide mid-point review: full model latency-vs-F1 table (RF → Transformer), deployment-feasible models highlighted; rubric scored                             |
| 5    | `W05_RL_Foundations_Report.md`      | 3-page: MDP formulation and reward design, value-based RL results, offline RL discussion, PIC 2.0 GRPO connection                                                  |
| 6    | `W06_RL_Advanced_Report.md`         | 4-page: policy-gradient vs value-based, GRPO-style result, IRL/hierarchical, multi-agent, deployment latency, HTD-IRL & CRL-MRS connections                        |
| 7    | `W07_Methodology_Report.md`         | 5-page publication-standard methodology: preprocessing, supervised comparison, sequence/attention, RL evaluation protocol, reproducibility, deployment feasibility |
| 7    | `W07_PIC20_ML_Analysis.md`          | 4–5 page PIC 2.0 model-class analysis: per-class experimental finding, methodology, deployment-readiness score (1–5), priority next experiment                   |
| 8    | `W08_Capstone_Report.docx`          | 18–22 page technical ML analysis report (full methodology across all families)                                                                                    |
| 8    | `W08_Capstone_Deck.pptx`            | 12-slide executive deck with finding-statement headlines (specific numbers on every slide)                                                                         |
| 8    | `W08_Retrospective.md`              | 1-page retrospective: most surprising finding, one design decision to change, thesis connection spanning trajectory-prediction and RL-control framings             |

## Toolchain

| Category                       | Tools                                                                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Core Python & Data             | Python 3.11, pandas, NumPy, SciPy (scipy.fft, scipy.stats)                                                                  |
| Classical ML                   | scikit-learn (RF, PCA, DT, SVC, GaussianNB, KMeans, learning_curve)                                                         |
| Neural Networks & Transformers | PyTorch (MLP, Conv1d, LSTM, GRU, TransformerEncoder, Transformer seq2seq, quantization)                                    |
| Reinforcement Learning         | Gymnasium (env API), Stable-Baselines3 (DQN, PPO), PettingZoo (multi-agent); from-scratch REINFORCE / GRPO-style in PyTorch |
| Explainability                 | SHAP (TreeExplainer, DeepExplainer, KernelExplainer); attention-map extraction; RL value/saliency views                     |
| Evaluation & Benchmarking      | scikit-learn metrics, timeit, pytest; RL return mean ± std over ≥5 seeds                                                  |
| Visualisation                  | matplotlib, seaborn (heatmaps, ROC curves, SHAP plots, attention maps, learning curves, RL return bands)                    |

## Model × Task Mapping

| Model / Algorithm     | Feature Selection | Anomaly Detection | Trajectory Prediction | RL / Control | Explainability | Pareto |
| --------------------- | :---------------: | :---------------: | :-------------------: | :----------: | :------------: | :----: |
| FFT                   |        ✓        |                  |                      |              |                |        |
| PCA                   |        ✓        |                  |                      |              |                |        |
| Random Forest         |        ✓        |        ✓        |                      |              |   SHAP Tree   |   ✓   |
| MLP                   |                  |        ✓        |          ✓          |              |   SHAP Deep   |   ✓   |
| 1D-CNN                |                  |        ✓        |                      |              |   SHAP Deep   |   ✓   |
| LSTM                  |                  |        ✓        |          ✓          |              |   SHAP Deep   |   ✓   |
| GRU                   |                  |        ✓        |                      |              |                |   ✓   |
| Transformer Encoder   |                  |        ✓        |                      |              | Attention Maps |   ✓   |
| Transformer seq2seq   |                  |                  |          ✓          |              | Attention Maps |   ✓   |
| DQN                   |                  |                  |                      |      ✓      |                |   ✓   |
| Offline RL / BC       |                  |                  |                      |      ✓      |                |   ✓   |
| PPO                   |                  |                  |                      |      ✓      |  RL Saliency  |   ✓   |
| GRPO-style            |                  |                  |                      |      ✓      |  RL Saliency  |   ✓   |
| IRL                   |                  |                  |                      |   <br />✓   |                |        |
| Multi-Agent (CRL-MRS) |                  |                  |                      |      ✓      |                |        |

## Quick Start

### 1. Initialize repo

```bash
mkdir -p ~/MY_WORKSPACE && cd ~/MY_WORKSPACE
git clone https://github.com/LHY3015/hongyu-ingen-ml-analyst.git
cd hongyu-ingen-ml-analyst
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```


```bash
# Install Pytorch and CUDA that matches your hardware (see https://pytorch.org/get-started/locally/):
# Replace cuXXX with your version, e.g. cu130 for CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
```

### 3. Run

Open any `notebooks/WXX_*.ipynb` and run all cells. Random seeds are documented at the top of each notebook.
