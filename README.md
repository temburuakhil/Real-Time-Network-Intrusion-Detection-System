# Lightweight Network Intrusion Detection System (14-Feature NIDS)

A high-performance, real-time Network Intrusion Detection System (NIDS) utilizing the **UNSW-NB15 dataset** to demonstrate that cyberattacks can be accurately detected using only **14 selected features** instead of the typical 42+ features.

---

## 📌 Executive Summary & Core Research Goal
Traditional intrusion detection systems process 40+ packet features, which introduces substantial computational overhead and latency, making real-time wire-speed inspection difficult. 

By utilizing **Recursive Feature Elimination (RFECV)** (Awad & Fraihat, 2023), this project reduces the feature space down to **14 key traffic and flow features**, achieving:
* **98.35% Attack Recall** with **Random Forest** (detecting 44,584 out of 45,332 attacks in the test set).
* Sub-2-second training time over 175,000+ flow records.
* Complete deployment pipeline with an interactive **Streamlit** dashboard.

---

## 🎯 The Exact 14 Features

| # | Feature Name | Category | Description | Cybersecurity Significance |
|---|---|---|---|---|
| 1 | `rate` | Flow | Transferred packets per second | Volumetric attack detection (e.g. DoS/DDoS flood) |
| 2 | `sttl` | Header | Source to destination Time to Live | Identifies distinct OS signatures and scanner TTL footprints |
| 3 | `sload` | Flow | Source bits per second | Source transmission bandwidth; identifies high-load exfiltration |
| 4 | `dload` | Flow | Destination bits per second | Destination response throughput; near-zero indicates target starvation |
| 5 | `ct_srv_src` | Connection | Same service & source address in 100 connections | Identifies repeated automated scanner requests to a single service |
| 6 | `ct_state_ttl` | Connection | Same state and dst/src TTL values | Synthetic or scripted packet generator signature |
| 7 | `ct_dst_ltm` | Connection | Connections to the same destination address | Concentrated attack against a specific destination IP |
| 8 | `ct_src_dport_ltm` | Connection | Connections from same source to same dest port | Targeted port attacks (e.g. brute force on SSH/HTTP) |
| 9 | `ct_dst_sport_ltm` | Connection | Distinct destination ports from source IP | Port-scanning behavior (probing multiple service ports) |
| 10 | `ct_dst_src_ltm` | Connection | Connections between same source & destination IP | High interaction intensity between host pairs |
| 11 | `ct_src_ltm` | Connection | Total connections from same source IP | Overall localized activity of source host |
| 12 | `ct_srv_dst` | Connection | Connections to same service & destination IP | Service-specific overloading or denial of service |
| 13 | `state_CON` | Categorical (Encoded) | Connection state is `CON` (Connected) | Legitimate two-way session indicator |
| 14 | `state_INT` | Categorical (Encoded) | Connection state is `INT` (Intermediate) | Incomplete / half-open scans and connectionless floods |

* **Target Label (`label`):** `0` = Normal, `1` = Attack.

---

## 📊 Experimental Results (Evaluated on 82,332 Test Samples)

| Model | Accuracy | Precision | Recall (Detection Rate) | F1-Score | ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **85.91%** | **80.42%** | **98.35%** | **0.8849** | **0.9572** | **1.78 s** |
| **AdaBoost** | 81.87% | 76.26% | 97.39% | 0.8554 | 0.9419 | 4.66 s |
| **Multi-Layer Perceptron (MLP)** | 81.27% | 76.41% | 95.45% | 0.8488 | 0.9271 | 30.86 s |
| **Stochastic Gradient Descent (SGD)** | 75.93% | 71.99% | 92.13% | 0.8083 | 0.8718 | 0.19 s |
| **Logistic Regression** | 75.31% | 71.46% | 91.83% | 0.8038 | 0.8756 | 0.41 s |
| **Naive Bayes (Gaussian)** | 75.00% | 85.42% | 65.83% | 0.7435 | 0.7932 | 0.04 s |

---

## 📁 Project Structure

```text
Saq Project/
│
├── Training and Testing Sets/
│   ├── UNSW_NB15_training-set.csv     # 175,341 training records
│   └── UNSW_NB15_testing-set.csv      # 82,332 testing records
│
├── src/
│   ├── preprocess.py                  # 14-feature extraction & MinMaxScaler
│   └── train_evaluate.py              # 6-model training, evaluation & metrics
│
├── models/
│   ├── scaler.joblib                  # Serialized MinMaxScaler
│   ├── random_forest.joblib           # Trained Random Forest model
│   ├── adaboost.joblib                # Trained AdaBoost model
│   ├── multi_layer_perceptron.joblib  # Trained MLP Neural Network model
│   ├── logistic_regression.joblib     # Trained Logistic Regression model
│   ├── stochastic_gradient_descent.joblib # Trained SGD model
│   └── naive_bayes.joblib             # Trained Naive Bayes model
│
├── results/
│   ├── model_comparison.csv          # Evaluation metrics CSV
│   ├── model_comparison.json         # Evaluation metrics JSON
│   ├── model_comparison_chart.png    # Bar chart of metric comparisons
│   └── roc_curves.png                # ROC Curves plot
│
├── data/
│   └── sample_test_packets.csv       # Pre-extracted 200 packets for demo
│
├── app.py                             # Interactive Streamlit Web Application
├── run_app.bat                        # One-click launcher for the web app
└── README.md                          # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Run Model Training & Evaluation
To retrain the 6 models or re-evaluate metrics:
```bash
python src/train_evaluate.py
```

### 2. Launch the Interactive Web Dashboard
Run via command line:
```bash
streamlit run app.py
```
Or double-click the `run_app.bat` file in Windows Explorer.

---

## 🖥️ Web Application Features

1. **Real-Time Packet Inspector:**
   * Enter values for the 14 features or click preset scenario buttons:
     * 🟢 *Normal Web Traffic*
     * 🔴 *DDoS / Rate Flood Attack*
     * 🔴 *Reconnaissance Scan Attack*
   * Displays immediate classification (Normal vs Attack) and prediction confidence gauge.
2. **Batch Log Analyzer:**
   * Drag-and-drop any network capture CSV or load the pre-packaged 200-sample test batch.
   * View live statistics, attack percentages, visual breakdown bar chart, and download the tagged CSV.
3. **Model Benchmark & Comparison:**
   * Interactive table and high-resolution charts for ROC Curves and performance metrics.
4. **14-Feature Documentation:**
   * Comprehensive reference guide explaining the role of each feature in intrusion detection.

---

## 🎓 Academic Alignment & Presentation Guide

This project matches the presentation and report guidelines:
* **Objective:** Real-time intrusion detection via 14-feature reduction.
* **Literature Foundation:** Awad & Fraihat (2023), Moustafa & Slay (2015).
* **Comparative Study:** 6 ML models compared across Accuracy, Precision, Recall, F1, and Latency.
* **Reproducibility:** Seed `random_state=42` used uniformly.
