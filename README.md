# lstm-drug-review_adv

# Vanilla and Gated LSTM Variants for Drug-Review Sentiment Classification

**Advanced Natural Language Processing: Lab Project on Recurrent Neural Networks**
International University of Rabat (UIR), School of Data Science and Digital Engineering, Data Engineering Program (5th year), 2026-2027

![PyTorch](https://img.shields.io/badge/PyTorch-from%20scratch-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/task-3--class%20sentiment-1B2A4A)
![Dataset](https://img.shields.io/badge/data-Drugs.com%20(UCI%20%23462)-2E7D32)
![Metric](https://img.shields.io/badge/main%20metric-macro--F1-F58518)

| | |
|---|---|
| **Team** | MohamedTahaELQASSID

---

## Table of contents
1. [Overview](#1-overview)
2. [Key results](#2-key-results)
3. [Data and preprocessing](#3-data-and-preprocessing)
4. [Models](#4-models)
5. [Experiments](#5-experiments)
6. [Final model and error analysis](#6-final-model-and-error-analysis)
7. [Additional analyses](#7-additional-analyses)
8. [Limitations](#8-limitations)
9. [Repository structure](#9-repository-structure)
10. [How to run](#10-how-to-run)
11. [Reproducibility](#11-reproducibility)
12. [References and license](#12-references-and-license)

---

## 1. Overview
Patient drug reviews are short informal texts whose sentiment depends on word order. The task is to predict, from the review text, whether the 10-star rating is **negative** (1-4), **neutral** (5-6) or **positive** (7-10).

We implement a **vanilla LSTM** and three variants **from scratch in PyTorch** using `nn.Linear` (the built-in `nn.LSTM` is used only once, as a unit-test reference for the vanilla cell):

| Cell | Idea |
|---|---|
| **Vanilla** | Standard LSTM (baseline) |
| **WMC** | Working Memory Connections: peephole terms from the cell state into the gates |
| **CIFG** | Coupled Input-Forget Gate: `i_t = 1 - f_t` (one gate fewer) |
| **LSTMP** | LSTM with a projection layer `h_t = W_p m_t + b_p` (`P < H`) |

All models are compared with identical splits and protocol, with parameter counts reported, and with an additional **matched-parameter** study.

Because the classes are imbalanced (66% positive), the main metric is the **macro-averaged F1**; accuracy is secondary.

## 2. Key results

**Main findings**
- Scaling the training set from 3k to 60k reviews raised macro-F1 by about **+0.15** for every cell, roughly ten times the largest gap between cells.
- With three seeds, the four cells are statistically indistinguishable (spread 0.014 vs. seed std up to 0.016), so **we do not rank them**.
- CIFG is the cheapest cell (fewest parameters, 14% faster per epoch than vanilla); WMC is the most expensive (about 23% slower) for a gain within noise.
- Learning rate, dropout, batch size and class weighting matter more than optimizer family or initialization.
- The **neutral class** is the bottleneck for every model (recall about 0.2); class weighting roughly doubles it at the cost of positive recall.

**Test macro-F1 (mean ± std over 3 seeds, official test set)**

| Cell | Total params | Exp. A (H = 128) | Exp. B (matched recurrent params) | s/epoch (A) |
|---|---:|---:|---:|---:|
| Vanilla | 2,334,595 | 0.5820 ± 0.0029 | 0.5820 ± 0.0029 | 62.1 |
| WMC | 2,383,747 | 0.5964 ± 0.0115 | 0.5909 ± 0.0106 | 76.1 |
| CIFG | 2,301,699 | 0.5922 ± 0.0079 | 0.5983 ± 0.0163 | 53.5 |
| LSTMP | 2,309,891 | 0.5859 ± 0.0101 | 0.5961 ± 0.0132 | 66.8 |

**Final tuned model** (CIFG, batch size 64, up to 12 epochs, single run, seed 0): **accuracy 0.819, macro-F1 0.640**.
This run used more epochs and tuned settings than A and B, so it is *not* evidence that CIFG beats the other cells.

![Dashboard A vs B](results/A_vs_B/dashboard_A_B.png)

## 3. Data and preprocessing
- **Dataset:** Drug Review Dataset (Drugs.com), UCI Machine Learning Repository #462: 161,297 training and 53,766 test reviews with a 10-star rating. The data are **not included** in this repository (see [How to run](#10-how-to-run)).
- **Cleaning:** HTML-entity decoding, lowercasing, removal of non-alphanumeric symbols, whitespace tokenization.
- **Vocabulary:** built on the training split only (min. frequency 3, at most 30,000 words): 17,208 entries including `<pad>` and `<unk>`.
- **Sequences:** truncated to 128 tokens (about a quarter of the reviews are longer) and padded; a mask freezes the recurrent state on padding so the classifier uses the hidden state of the last real token.
- **Splits (same for every model):** stratified 90/10 split of the official training file, then stratified subsampling to **60,000 training** and **10,000 validation** reviews. The official **test set (53,766)** is used only for final numbers.

| Class | Ratings | Train share (full) | Train share (60k subset) | Test n | Test share |
|---|---|---:|---:|---:|---:|
| Negative | 1-4 | 24.8% | 24.8% | 13,497 | 25.1% |
| Neutral | 5-6 | 8.9% | 8.9% | 4,829 | 9.0% |
| Positive | 7-10 | 66.3% | 66.3% | 35,440 | 65.9% |

Always predicting "positive" gives about 66% accuracy but only about 0.265 macro-F1, which is why accuracy alone is misleading.

![EDA](results/eda/eda.png)
![Class shares](results/eda/subset_check.png)

## 4. Models
**Architecture (identical for all models):** embedding (128) → dropout → one recurrent layer (our own cell, unrolled over time) → final hidden state → dropout → `nn.Linear(3)`. The gate matrices are stacked into one `nn.Linear` of size `4H` (`3H` for CIFG), which is mathematically identical to separate `W` matrices but faster. The forget-gate bias is initialized to 1.

**Vanilla LSTM**
```
i_t = σ(W_xi x_t + W_hi h_{t-1} + b_i)      f_t = σ(W_xf x_t + W_hf h_{t-1} + b_f)
o_t = σ(W_xo x_t + W_ho h_{t-1} + b_o)      g_t = tanh(W_xg x_t + W_hg h_{t-1} + b_g)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t             h_t = o_t ⊙ tanh(c_t)
```

| Cell | Modification | Recurrent params (E = H = 128) |
|---|---|---:|
| Vanilla | baseline | 131,584 |
| WMC | `i_t, f_t` gain `tanh(W_ci c_{t-1})`, `tanh(W_cf c_{t-1})`; `o_t` gains `tanh(W_co c_t)` | 180,736 |
| CIFG | `i_t = 1 - f_t` | 98,688 |
| LSTMP | `h_t = W_p (o_t ⊙ tanh c_t) + b_p`, `P = 64` | 107,072 |

**Correctness check.** Weights of our vanilla cell were copied into `nn.LSTM` (test only) and outputs compared on random double-precision input: maximum absolute difference **exactly 0**. As a control, perturbing one recurrent weight by 1e-3 raised the difference to **3.4e-5**, so the test is sensitive to errors. The other three cells have no PyTorch reference.

## 5. Experiments

### Common settings (A and B)
Embedding 128, hidden size 128 (P = 64 for LSTMP), dropout 0.3, Adam with lr 1e-3, batch size 128, gradient clipping 1.0, at most 8 epochs (patience 3 on validation macro-F1), best validation epoch kept, seeds 0, 1, 2.

### Experiment A: same hidden size
| Cell | Val. F1 | Test acc. | Test macro-F1 |
|---|---:|---:|---:|
| Vanilla | 0.5808 ± 0.0042 | 0.8092 | 0.5820 ± 0.0029 |
| WMC | 0.5936 ± 0.0128 | 0.8150 | 0.5964 ± 0.0115 |
| CIFG | 0.5912 ± 0.0063 | 0.8145 | 0.5922 ± 0.0079 |
| LSTMP | 0.5831 ± 0.0076 | 0.8115 | 0.5859 ± 0.0101 |

### Experiment B: matched recurrent parameters
The hidden size of each variant is searched so that its recurrent parameter count matches the vanilla cell (131,584), within 1.4%.

| Cell | H | Recurrent params | Total params | Val. F1 | Test macro-F1 | s/epoch |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 128 | 131,584 | 2,334,595 | 0.5808 ± 0.0042 | 0.5820 ± 0.0029 | 62.1 |
| WMC | 106 | 133,348 | 2,336,293 | 0.5848 ± 0.0129 | 0.5909 ± 0.0106 | 70.0 |
| CIFG | 154 | 130,746 | 2,333,835 | 0.5974 ± 0.0126 | 0.5983 ± 0.0163 | 58.5 |
| LSTMP | 148 | 131,202 | 2,334,051 | 0.5890 ± 0.0070 | 0.5961 ± 0.0132 | 72.5 |

Effect of matching (test macro-F1, B minus A): WMC -0.0055, CIFG +0.0061, LSTMP +0.0102, all smaller than the seed std. Part of WMC's advantage in A may have come from its 49k extra parameters.

Per-seed test macro-F1 (the same configuration varies by up to about 3 points between seeds):

| Cell | A seed 0 | A seed 1 | A seed 2 | B seed 0 | B seed 1 | B seed 2 |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 0.5801 | 0.5853 | 0.5804 | = A | = A | = A |
| WMC | 0.6095 | 0.5877 | 0.5920 | 0.5810 | 0.6022 | 0.5894 |
| CIFG | 0.5836 | 0.5937 | 0.5992 | 0.5842 | 0.6161 | 0.5947 |
| LSTMP | 0.5859 | 0.5960 | 0.5757 | 0.5857 | 0.5917 | 0.6110 |

![Curves A](results/A/curvesA_60k.png)
![Curves B](results/B/curves_B.png)

### Effect of training-set size (3k pilot vs. 60k)
| Cell | Pilot (3k) | Exp. A (60k) | Gain |
|---|---:|---:|---:|
| Vanilla | 0.4261 | 0.5820 | +0.156 |
| WMC | 0.4440 | 0.5964 | +0.152 |
| CIFG | 0.4440 | 0.5922 | +0.148 |
| LSTMP | 0.4260 | 0.5859 | +0.160 |

### Experiment C: hyperparameters
One factor at a time, single seed, 5 epochs, **validation only** (vanilla and CIFG). Since the models are still under-trained at 5 epochs, the protocol favours settings that converge quickly.

| Trial (val. macro-F1) | Vanilla | CIFG |
|---|---:|---:|
| Baseline | 0.5382 | 0.5689 |
| lr 3e-4 | 0.4980 | 0.5090 |
| lr 3e-3 | 0.5647 | 0.5986 |
| AdamW (wd 0.01) | 0.5379 | 0.5722 |
| RMSprop | 0.5475 | 0.5612 |
| SGD (lr 0.1, momentum 0.9) | 0.5017 | 0.5165 |
| Batch 64 | 0.5622 | **0.6028** |
| Batch 256 | 0.5287 | 0.5287 |
| Xavier init | 0.5378 | 0.5592 |
| Orthogonal init | 0.5487 | 0.5550 |
| Dropout 0.1 | 0.5907 | 0.5934 |
| Dropout 0.5 | 0.5078 | 0.5292 |
| Class weights | **0.5961** | 0.5879 |

Helpful for both cells: dropout 0.1, higher learning rate, batch 64, class weights. Harmful for both: lr 3e-4, SGD, dropout 0.5. The effects of optimizer family and initialization change sign between the two cells and stay within the noise level (about 0.01-0.02).

![Hyperparameters](results/C/hparam_plot.png)

## 6. Final model and error analysis
The best single validation score in Experiment C was CIFG with batch size 64 (0.6028). It was retrained for up to 12 epochs (patience 4, seed 0) and the test set was evaluated **once**: accuracy **0.819**, macro-F1 **0.640** (best validation macro-F1 0.645; 2,301,699 parameters).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.788 | 0.721 | 0.753 | 13,497 |
| Neutral | 0.380 | 0.208 | 0.269 | 4,829 |
| Positive | 0.859 | 0.940 | 0.897 | 35,440 |
| **Macro avg** | 0.676 | 0.623 | **0.640** | 53,766 |
| Weighted avg | 0.798 | 0.819 | 0.805 | 53,766 |

**Findings**
- The **neutral class is the bottleneck**: 54% of neutral reviews are predicted positive and 25% negative; only 21% are recovered. Positive reviews are almost never predicted neutral (2%).
- Performance is **stable across review lengths** (macro-F1 0.627-0.640 for all four length buckets).
- Training loss falls from 0.714 to 0.323 while validation loss is lowest at epoch 5 (0.498) and rises to 0.558 at epoch 12; validation macro-F1 still improves until epoch 12, so early stopping on F1 did not trigger.

![Confusion matrix](results/final/confusion.png)
![Final curves](results/final/final_curves.png)

**Qualitative check** (saved model reloaded from disk; test metrics reproduced):

| Review | Predicted | neg | neu | pos |
|---|---|---:|---:|---:|
| "This medication changed my life, no side effects and my pain is gone." | positive | 0.00 | 0.00 | 1.00 |
| "Terrible. Nausea every day and it did nothing for my anxiety." | negative | 1.00 | 0.00 | 0.00 |
| "It worked a little but the headaches were annoying, I'm not sure it is worth it." | neutral | 0.34 | 0.49 | 0.18 |
| "Great at first but it stopped working and the side effects got worse." | negative | 0.49 | 0.25 | 0.27 |

These four examples are anecdotal and do not replace the test-set evaluation.

## 7. Additional analyses
Files in `results/extra/`.

**Tuned comparison (single runs, seed 0).** Vanilla trained with the same settings as the final model reaches a neutral recall similar to CIFG's (about 0.19 vs. 0.21) and a higher negative recall (about 0.81 vs. 0.72), so the weak neutral performance is a property of the setup and the task, not of the cell. CIFG with class weights raises the neutral recall to about 0.45 at the price of roughly 9 points of positive recall (0.94 to about 0.85). Recalls are read from the figure; exact values are in `results/extra/final_comparison.csv`.

![Per-class recall](results/extra/final_comparison.png)

**Forget-gate analysis (one seed, descriptive).** Over the first 50 tokens, CIFG's mean forget gate rises from about 0.73 to about 0.80 and stays there, while vanilla and LSTMP stay near 0.75 and WMC near 0.73. On one example review, vanilla, WMC and LSTMP show drops at the contrastive word "but", while CIFG's stays higher. This is anecdotal evidence from one seed and one review; no causal claim is made.

![Gate position](results/extra/gate_position.png)
![Gate tokens](results/extra/gate_tokens.png)

## 8. Limitations
- Only 3 seeds for A/B and a single seed for Experiment C and for the final model (seed-to-seed variation is about ±1 point; no confidence intervals or paired tests).
- Validation curves are still rising at epoch 8, so the models are under-trained in A and B.
- Hyperparameter trials are single 5-epoch runs; the final model received more epochs than A and B, so the final result is not comparable with them.
- 60,000 of the roughly 145,000 available training reviews were used; reviews are truncated to 128 tokens.
- The Experiment C table was transcribed from the notebook's printed output because the raw CSV could not be downloaded.

**Possible improvements:** class weighting or threshold tuning, or regression on the rating (ordinal labels); bidirectional layers with attention or max pooling; pretrained embeddings (e.g. fastText); full training set and longer schedules with learning-rate decay; more seeds with significance tests.



## 10. How to run
1. **Get the data:** the Kaggle copy of the UCI dataset ("UCI ML Drug Review Dataset", files `drugsComTrain_raw.csv` and `drugsComTest_raw.csv`), or download it from https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com.
2. **Create a Kaggle notebook**, add the dataset with *Add Input*, and enable a **GPU** accelerator.
3. **Upload `notebook.ipynb`** and run the cells in order.
4. **Quick test:** set `MAX_TRAIN = 3000` in the first cell (about 10 minutes). **Full run:** `MAX_TRAIN = 60000`.

Approximate full-run time on a Kaggle GPU: Experiment A about 1 h, B about 1 h, C about 2.5 h, final model about 20 min.
Re-run the data cell (Cell 4) whenever `MAX_TRAIN` changes; do not re-run the Experiment A cell after Experiment B (it resets the stored results).

**Requirements:** Python 3, PyTorch, pandas, NumPy, scikit-learn, matplotlib (all preinstalled in Kaggle notebooks).

## 11. Reproducibility
- Fixed seeds (0, 1, 2) and fixed split seed (`random_state = 42`); vocabulary built on the training split only.
- Test set used only for the A/B summaries and the final model; all hyperparameter choices were made on validation data.
- Small differences are expected across different GPUs and PyTorch versions.
- The trained final model is stored in `results/final/final_cifg.pt`:
```python
model = Classifier(len(itos), "cifg", 128, 128, 64, 1, 0.3)
model.load_state_dict(torch.load("results/final/final_cifg.pt", map_location="cpu"))
```

## 12. References and license
1. F. Grässer, S. Kallumadi, H. Malberg, S. Zaunseder, *Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning*, DH'18. Dataset: UCI Machine Learning Repository #462 (CC BY 4.0).
2. K. Greff et al., *LSTM: A Search Space Odyssey*, IEEE TNNLS, 2017 (arXiv:1503.04069): CIFG.
3. F. Landi et al., *Working Memory Connections for LSTM*, Neural Networks 144, 2021: WMC.
4. H. Sak, A. Senior, F. Beaufays, *Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling*, Interspeech 2014: LSTMP.

The dataset is licensed under CC BY 4.0 by its authors and is not redistributed here. Code and results: TODO choose a license (for example MIT) or state "academic project, all rights reserved".
