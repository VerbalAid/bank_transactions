# Outlier Detection — ML Coursework

A practical exploration of **unsupervised anomaly detection** across two real-world datasets. The notebook implements and compares three distinct outlier detection paradigms — tree-based isolation, density-based neighbourhood analysis, and neural network reconstruction error — applying each to both a classic spam dataset and a financial fraud dataset.

---

## 📋 Overview

| | Exercise 1 | Exercise 2 |
|---|---|---|
| **Dataset** | Spambase | Bank Transaction Fraud |
| **Task** | Detect anomalous Ham emails | Detect fraudulent transactions |
| **Models** | Isolation Forest, LOF, Autoencoder (sklearn) | Z-score baseline, Isolation Forest, LOF, Autoencoder (Keras) |
| **Visualisation** | Histogram, PCA circle plot, PCA scatter, t-SNE | Histogram, scatter, LOF circle plot, index plot, age demographics |

---

## 🗂️ Exercise 1 — Spambase Dataset

### Dataset
The [Spambase dataset](http://www.sc.ehu.es/ccwbayes/master/selected-dbs/nlp-naturallanguageprocessing/spambase.csv) contains 4,601 emails pre-labelled as Spam (`1`) or Ham (`0`), described by 57 features covering word frequencies, character frequencies (e.g. `$`, `!`), and capital letter run statistics.

**Key methodological choice:** only Ham emails are used for fitting. By removing the class label and training exclusively on non-spam, the models are forced to learn what "normal" looks like — any email that deviates from that learned normality is flagged as an outlier. This mirrors a real-world unsupervised deployment where labels don't exist.

---

### Model 1 — Isolation Forest

Isolation Forest works by randomly partitioning the feature space using decision trees. Normal points require many splits to isolate; anomalies, being rare and different, are isolated in very few splits. The anomaly score is the inverse of the average path length across the ensemble.

- **Contamination:** 5%
- **Score convention:** `-decision_function()` is used so that higher scores = more anomalous
- **Visualisation:** Histogram of anomaly scores with a threshold line at 0

---

### Model 2 — Local Outlier Factor (LOF)

LOF compares the local density of a point to the densities of its `k` nearest neighbours. A point surrounded by a much sparser neighbourhood than its neighbours receives a high LOF score. Unlike Isolation Forest, LOF is sensitive to *local* context — it can catch anomalies that look normal globally but are unusual for their specific region of feature space.

- **Neighbours:** `n_neighbors=20`, **Contamination:** 5%
- **Visualisation:** PCA-reduced 2D scatter with circle sizes proportional to LOF score — large circles indicate high outlier factor

---

### Model 3 — Autoencoder (Neural Network)

An autoencoder is trained to compress and reconstruct the input features through a narrow bottleneck layer. Because it trains only on Ham emails, it learns the structure of "normal" email content. When presented with an unusual email, the reconstruction is poor — the Mean Squared Error (MSE) between input and output serves as the anomaly score.

- **Architecture:** 57 → 10 → 57 via `sklearn.neural_network.MLPRegressor`
- **Threshold:** 95th percentile of reconstruction errors
- **Visualisations:**
  - Histogram with KDE of reconstruction errors
  - PCA scatter coloured by MSE (`magma` colormap)
  - t-SNE scatter coloured by MSE — noted as superior to PCA for revealing cluster structure and isolating fringe points

---

## 🏦 Exercise 2 — Bank Transaction Fraud Detection

### Dataset
[Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection) — 2,500 transaction records with 16 features including `TransactionAmount`, `TransactionDuration`, `LoginAttempts`, `AccountBalance`, and `CustomerAge`. No missing values.

### Feature Selection
Five features were selected based on domain reasoning about fraud indicators:

| Feature | Fraud Rationale |
|---|---|
| `TransactionAmount` | Unusually high amounts signal fraud |
| `TransactionDuration` | Abnormal timing may indicate bot activity |
| `LoginAttempts` | High attempts suggest brute force access |
| `AccountBalance` | Sudden drops indicate drained accounts |
| `CustomerAge` | Age mismatches may indicate identity spoofing |

All features were standardised with `StandardScaler` before modelling.

---

### Statistical Baseline — Z-score Analysis
Before any ML modelling, a Z-score baseline was established by flagging transactions where `TransactionAmount` exceeded 3 standard deviations from the mean. This yielded **48 baseline outliers** and serves as a reference point: detections that go beyond simple magnitude thresholding indicate the models are capturing more complex behavioural patterns.

---

### Model 1 — Isolation Forest
- **Contamination:** 1% (realistic fraud rate assumption)
- **Result:** 26 anomalies — fewer than the Z-score baseline, confirming the model captures multi-feature patterns rather than just extreme amounts
- **Visualisation:** Anomaly score histogram + `TransactionAmount` vs `LoginAttempts` scatter with detected fraud highlighted in red

**Key finding:** High login attempt counts (4+) are rare in the dataset; most normal transactions clear in 1–3 attempts, making persistent login patterns an easy isolation target.

---

### Model 2 — Local Outlier Factor (LOF)
- **Neighbours:** `n_neighbors=20`, **Contamination:** 1%
- **Visualisation:** Circle plot on `TransactionAmount` vs `LoginAttempts` with circle radius proportional to normalised LOF score

**Key finding:** The most extreme outliers (largest circles) occurred at *2* login attempts — not 4 or 5. These points are highly isolated from their local neighbours despite not having the highest raw attempt count, demonstrating LOF's ability to surface local density anomalies that global methods miss.

**Model consensus check:** Transactions flagged by *both* Isolation Forest and LOF were extracted as high-certainty fraud candidates. The intersection highlighted account `AC00083` — substantiated by manual inspection showing a location change (Mesa/Detroit → San Diego), a device swap, 4 login attempts, and an account balance drop to $859.86 from a usual range of $3,000–$6,000.

---

### Model 3 — Autoencoder (Keras/TensorFlow)
- **Architecture:** 5 → 8 → 4 → 8 → 5 (encoder-decoder with a 4-dimensional bottleneck)
- **Training:** 50 epochs, Adam optimiser, MSE loss, 10% validation split
- **Threshold:** 99th percentile of reconstruction errors
- **Visualisation:** Per-transaction MSE index plot with 99th percentile threshold line

**Deep-dive — Index 2149 (Reconstruction Error: 2.97):**
The most extreme anomaly in the dataset. Compared to the account owner's historical baseline:
- Account balance $6,463 above the owner's norm
- Transaction amount $957 above average
- Different device used via Online channel
- 4 login attempts vs. the owner's typical 1

**Deep-dive — Index 993:**
A subtler case flagged due to the co-occurrence of $266 over-spend, 76 seconds longer transaction duration, and a 31-year age discrepancy from the typical customer — a combination of features that rarely appears in normal transactions.

---

### Age Mismatch / Identity Spoofing Analysis
An extended investigation into `CustomerAge` across detected anomalies revealed a counterintuitive pattern: the 18–30 age group showed the highest raw anomaly count, but when the transaction-time age was compared against each account's historical average age, large mismatches emerged — e.g. a transaction logged as age 22 from an account belonging to a 55-year-old.

**Conclusion:** The 18–30 spike is "Identity Noise." Scammers appear to inject younger ages during fraudulent transactions to avoid heuristic detection. The true victim demographic (based on historical account ages) concentrates in the 31–60 range. The Autoencoder flags these transactions precisely because the submitted age conflicts with the learned profile of the account.

---

## 📊 Visualisation Summary

| Plot | Purpose |
|---|---|
| Pie chart — Spam vs Ham | Class balance check |
| Histogram — IF anomaly scores | Score distribution with threshold at 0 |
| LOF circle plot (PCA 2D) | Local density deviation, circle size = outlier score |
| Histogram + KDE — AE reconstruction error | Error distribution with 95th percentile threshold |
| PCA scatter — AE error (magma) | Spatial mapping of reconstruction error |
| t-SNE scatter — AE error (magma) | Non-linear dimensionality reduction for clearer cluster separation |
| IF score histogram — bank data | 1% anomaly cutoff on transaction scores |
| IF scatter — Amount vs Login Attempts | Detected fraud highlighted in red |
| LOF circle plot — bank data | Normalised outlier scores in feature space |
| AE index plot — bank data | Per-transaction MSE with 99th percentile line |
| Age group bar charts | Apparent vs. true victim demographics |

---

## 🛠️ Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow scipy
```

The bank transaction dataset (`bank_transactions_data_2.csv`) must be downloaded separately from Kaggle and placed in the working directory (or `/content/` on Colab).

---

## 📄 Data Sources

| Dataset | Source |
|---|---|
| Spambase | [UPV/EHU course data](http://www.sc.ehu.es/ccwbayes/master/selected-dbs/nlp-naturallanguageprocessing/spambase.csv) |
| Bank Transaction Fraud | [Kaggle — valakhorasani](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection) |
