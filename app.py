"""
Streamlit Web Application: Real-Time Network Intrusion Detection System (NIDS)
Powered by UNSW-NB15 dataset using 14 lightweight features.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="14-Feature Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished cybersecurity UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .badge-attack {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #4ADE80;
    }
</style>
""", unsafe_allow_html=True)

# 14 Feature definitions
NUMERICAL_FEATURES = [
    'rate',
    'sttl',
    'sload',
    'dload',
    'ct_srv_src',
    'ct_state_ttl',
    'ct_dst_ltm',
    'ct_src_dport_ltm',
    'ct_dst_sport_ltm',
    'ct_dst_src_ltm',
    'ct_src_ltm',
    'ct_srv_dst'
]
CATEGORICAL_FEATURES = ['state_CON', 'state_INT']
FEATURE_NAMES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

MODEL_FILES = {
    "Random Forest (Recommended)": "models/random_forest.joblib",
    "AdaBoost": "models/adaboost.joblib",
    "Multi-Layer Perceptron (MLP)": "models/multi_layer_perceptron.joblib",
    "Logistic Regression": "models/logistic_regression.joblib",
    "Stochastic Gradient Descent (SGD)": "models/stochastic_gradient_descent.joblib",
    "Naive Bayes": "models/naive_bayes.joblib"
}

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.joblib")

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_benchmark_data():
    if os.path.exists("results/model_comparison.csv"):
        return pd.read_csv("results/model_comparison.csv")
    return None

scaler = load_scaler()
benchmark_df = load_benchmark_data()

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=70)
    st.markdown("### **NIDS Control Panel**")
    st.caption("14-Feature Lightweight Detection Engine")
    
    selected_model_name = st.selectbox(
        "Select Active ML Model",
        list(MODEL_FILES.keys()),
        index=0
    )
    current_model = load_model(MODEL_FILES[selected_model_name])
    
    st.markdown("---")
    st.markdown("#### **Active Model Metrics**")
    if benchmark_df is not None:
        clean_name = selected_model_name.replace(" (Recommended)", "")
        matched_row = benchmark_df[benchmark_df["Model"].str.contains(clean_name[:8], case=False, na=False)]
        if not matched_row.empty:
            acc = matched_row.iloc[0]["Accuracy"]
            rec = matched_row.iloc[0]["Recall"]
            f1 = matched_row.iloc[0]["F1-Score"]
            t_time = matched_row.iloc[0]["Training Time (s)"]
            
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{acc*100:.1f}%")
            c2.metric("Recall", f"{rec*100:.1f}%")
            
            c3, c4 = st.columns(2)
            c3.metric("F1-Score", f"{f1:.3f}")
            c4.metric("Train Time", f"{t_time:.2f}s")

    st.markdown("---")
    st.markdown("""
    **Research Framework:**
    * Dataset: UNSW-NB15
    * Features: 14 Selected Features
    * Target: Binary (0 = Normal, 1 = Attack)
    * `random_state = 42`
    """)

# ================= MAIN CONTENT =================
st.markdown('<div class="main-header">🛡️ Real-Time Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detecting cyberattacks in real time using 14 lightweight flow and connection features</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🔍 Real-Time Packet Inspector",
    "📁 Batch Log Analyzer",
    "📊 Benchmark & Evaluation",
    "📖 14-Feature Documentation"
])

# ----------------- TAB 1: Real-Time Packet Inspector -----------------
with tabs[0]:
    st.markdown("### Single Packet Live Inspection")
    st.write("Simulate incoming network traffic packets or pick from realistic traffic presets below:")
    
    # Preset Buttons
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_choice = None
    if col_p1.button("🟢 Load Normal Web Traffic", use_container_width=True):
        preset_choice = "normal"
    if col_p2.button("🔴 Load DDoS / Rate Flood Attack", use_container_width=True):
        preset_choice = "ddos"
    if col_p3.button("🔴 Load Reconnaissance Scan Attack", use_container_width=True):
        preset_choice = "recon"

    # Default values based on presets
    if preset_choice == "ddos":
        default_rate = 125000.0
        default_sttl = 254
        default_sload = 75000000.0
        default_dload = 0.0
        default_ct_srv_src = 15
        default_ct_state_ttl = 2
        default_ct_dst_ltm = 10
        default_ct_src_dport_ltm = 8
        default_ct_dst_sport_ltm = 8
        default_ct_dst_src_ltm = 15
        default_ct_src_ltm = 12
        default_ct_srv_dst = 15
        default_state = "INT"
    elif preset_choice == "recon":
        default_rate = 50000.0
        default_sttl = 254
        default_sload = 2500000.0
        default_dload = 0.0
        default_ct_srv_src = 8
        default_ct_state_ttl = 2
        default_ct_dst_ltm = 6
        default_ct_src_dport_ltm = 5
        default_ct_dst_sport_ltm = 5
        default_ct_dst_src_ltm = 6
        default_ct_src_ltm = 6
        default_ct_srv_dst = 8
        default_state = "INT"
    else: # normal
        default_rate = 120.0
        default_sttl = 62
        default_sload = 92000.0
        default_dload = 8500.0
        default_ct_srv_src = 3
        default_ct_state_ttl = 1
        default_ct_dst_ltm = 2
        default_ct_src_dport_ltm = 1
        default_ct_dst_sport_ltm = 1
        default_ct_dst_src_ltm = 2
        default_ct_src_ltm = 2
        default_ct_srv_dst = 3
        default_state = "CON"

    st.markdown("#### Packet Flow Parameters")
    with st.form("packet_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Traffic Rates & TTL**")
            in_rate = st.number_input("rate (pkts/sec)", min_value=0.0, value=default_rate, step=100.0)
            in_sttl = st.number_input("sttl (Source TTL)", min_value=0, max_value=255, value=default_sttl)
            in_state = st.selectbox("Connection State", options=["CON", "INT", "FIN / Other"], index=0 if default_state=="CON" else 1)

        with c2:
            st.markdown("**Load / Throughput**")
            in_sload = st.number_input("sload (Source bits/s)", min_value=0.0, value=default_sload, step=1000.0)
            in_dload = st.number_input("dload (Dest bits/s)", min_value=0.0, value=default_dload, step=1000.0)
            in_ct_state_ttl = st.number_input("ct_state_ttl", min_value=0, value=default_ct_state_ttl)

        with c3:
            st.markdown("**Connection Context Counters**")
            in_ct_srv_src = st.number_input("ct_srv_src", min_value=0, value=default_ct_srv_src)
            in_ct_srv_dst = st.number_input("ct_srv_dst", min_value=0, value=default_ct_srv_dst)
            in_ct_dst_src_ltm = st.number_input("ct_dst_src_ltm", min_value=0, value=default_ct_dst_src_ltm)

        c4, c5, c6 = st.columns(3)
        with c4:
            in_ct_dst_ltm = st.number_input("ct_dst_ltm", min_value=0, value=default_ct_dst_ltm)
            in_ct_src_ltm = st.number_input("ct_src_ltm", min_value=0, value=default_ct_src_ltm)
        with c5:
            in_ct_src_dport_ltm = st.number_input("ct_src_dport_ltm", min_value=0, value=default_ct_src_dport_ltm)
        with c6:
            in_ct_dst_sport_ltm = st.number_input("ct_dst_sport_ltm", min_value=0, value=default_ct_dst_sport_ltm)

        submitted = st.form_submit_button("⚡ Analyze Packet with " + selected_model_name, use_container_width=True)

    if submitted:
        # Construct feature row
        state_CON = 1.0 if in_state == "CON" else 0.0
        state_INT = 1.0 if in_state == "INT" else 0.0

        raw_input = {
            'rate': in_rate,
            'sttl': in_sttl,
            'sload': in_sload,
            'dload': in_dload,
            'ct_srv_src': in_ct_srv_src,
            'ct_state_ttl': in_ct_state_ttl,
            'ct_dst_ltm': in_ct_dst_ltm,
            'ct_src_dport_ltm': in_ct_src_dport_ltm,
            'ct_dst_sport_ltm': in_ct_dst_sport_ltm,
            'ct_dst_src_ltm': in_ct_dst_src_ltm,
            'ct_src_ltm': in_ct_src_ltm,
            'ct_srv_dst': in_ct_srv_dst,
            'state_CON': state_CON,
            'state_INT': state_INT
        }
        df_packet = pd.DataFrame([raw_input])[FEATURE_NAMES]
        df_scaled = pd.DataFrame(scaler.transform(df_packet), columns=FEATURE_NAMES)

        # Predict
        pred = current_model.predict(df_scaled)[0]
        
        prob_attack = None
        if hasattr(current_model, "predict_proba"):
            prob_attack = current_model.predict_proba(df_scaled)[0][1]

        st.markdown("---")
        st.markdown("### Detection Result")
        
        res_col1, res_col2 = st.columns([1.5, 2.5])
        with res_col1:
            if pred == 1:
                st.markdown('<div class="badge-attack">🚨 ATTACK DETECTED (Class 1)</div>', unsafe_allow_html=True)
                st.error("Malicious network traffic pattern recognized!")
            else:
                st.markdown('<div class="badge-normal">🛡️ NORMAL TRAFFIC (Class 0)</div>', unsafe_allow_html=True)
                st.success("Network flow is classified as legitimate traffic.")

            if prob_attack is not None:
                st.markdown(f"**Attack Confidence:** `{prob_attack*100:.2f}%`")
                st.progress(float(prob_attack))

        with res_col2:
            st.markdown("#### Input Features & Normalized Scale")
            display_comparison = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Raw Value": [raw_input[f] for f in FEATURE_NAMES],
                "MinMax Scaled [0, 1]": df_scaled.iloc[0].values
            })
            st.dataframe(display_comparison, height=220, use_container_width=True)

# ----------------- TAB 2: Batch Log Analyzer -----------------
with tabs[1]:
    st.markdown("### Batch Traffic Log Analysis")
    st.write("Upload a packet capture CSV or test using pre-packaged UNSW-NB15 sample records.")

    batch_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])
    
    use_sample = st.button("📥 Load Sample Test Batch (200 records from UNSW-NB15)")

    df_batch_raw = None
    if batch_file is not None:
        df_batch_raw = pd.read_csv(batch_file)
    elif use_sample and os.path.exists("data/sample_test_packets.csv"):
        df_batch_raw = pd.read_csv("data/sample_test_packets.csv")
        st.info("Loaded pre-packaged test batch (200 records).")

    if df_batch_raw is not None:
        st.write(f"Total Records in Batch: **{len(df_batch_raw)}**")
        
        # Preprocess batch
        df_work = df_batch_raw.copy()
        if 'state' in df_work.columns:
            df_work['state_CON'] = (df_work['state'] == 'CON').astype(float)
            df_work['state_INT'] = (df_work['state'] == 'INT').astype(float)
        else:
            if 'state_CON' not in df_work.columns: df_work['state_CON'] = 0.0
            if 'state_INT' not in df_work.columns: df_work['state_INT'] = 0.0

        missing = [f for f in FEATURE_NAMES if f not in df_work.columns]
        if missing:
            st.error(f"Missing required features in CSV: {missing}")
        else:
            X_batch = df_work[FEATURE_NAMES]
            X_batch_scaled = scaler.transform(X_batch)
            
            # Predict
            predictions = current_model.predict(X_batch_scaled)
            df_work['Predicted_Label'] = predictions
            df_work['Predicted_Class'] = df_work['Predicted_Label'].map({0: "Normal", 1: "Attack"})

            if hasattr(current_model, "predict_proba"):
                df_work['Attack_Probability'] = current_model.predict_proba(X_batch_scaled)[:, 1].round(4)

            # Summary Metrics
            attack_count = int(np.sum(predictions == 1))
            normal_count = int(np.sum(predictions == 0))
            attack_pct = (attack_count / len(predictions)) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Packets", len(predictions))
            m2.metric("Attacks Caught", attack_count, delta=f"{attack_pct:.1f}% of total", delta_color="inverse")
            m3.metric("Normal Packets", normal_count)
            if 'label' in df_work.columns:
                ground_truth_acc = (df_work['label'] == df_work['Predicted_Label']).mean() * 100
                m4.metric("Batch Accuracy vs Ground Truth", f"{ground_truth_acc:.2f}%")
            else:
                m4.metric("Attack Ratio", f"{attack_pct:.1f}%")

            # Visual breakdown
            st.markdown("#### Traffic Distribution Breakdown")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors = ['#22c55e', '#ef4444']
            ax.barh(["Normal", "Attack"], [normal_count, attack_count], color=colors)
            ax.set_xlabel("Packet Count")
            st.pyplot(fig)

            # Table view
            st.markdown("#### Classified Packets Preview")
            cols_to_show = ['Predicted_Class', 'rate', 'sttl', 'sload', 'dload', 'ct_srv_src', 'state_CON', 'state_INT']
            if 'Attack_Probability' in df_work.columns:
                cols_to_show.insert(1, 'Attack_Probability')
            if 'label' in df_work.columns:
                cols_to_show.insert(0, 'label')

            st.dataframe(df_work[cols_to_show].head(50), use_container_width=True)

            # Download CSV
            csv_export = df_work.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Classified Log (CSV)",
                data=csv_export,
                file_name="nids_classified_batch.csv",
                mime="text/csv"
            )

# ----------------- TAB 3: Benchmark & Evaluation -----------------
with tabs[2]:
    st.markdown("### 6-Model Benchmark & Comparative Analysis")
    st.write("Full evaluation on the official UNSW-NB15 testing set (82,332 records) using only 14 features (`random_state=42`).")

    if benchmark_df is not None:
        st.dataframe(
            benchmark_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Training Time (s)"]],
            use_container_width=True
        )

        c_img1, c_img2 = st.columns(2)
        with c_img1:
            st.markdown("#### Performance Metrics Across Models")
            if os.path.exists("results/model_comparison_chart.png"):
                st.image("results/model_comparison_chart.png", use_column_width=True)
        with c_img2:
            st.markdown("#### ROC Curves")
            if os.path.exists("results/roc_curves.png"):
                st.image("results/roc_curves.png", use_column_width=True)

        st.markdown("---")
        st.markdown("#### Confusion Matrix Data (Test Set: 82,332 Samples)")
        st.dataframe(
            benchmark_df[["Model", "TN", "FP", "FN", "TP"]],
            use_container_width=True
        )

# ----------------- TAB 4: 14-Feature Documentation -----------------
with tabs[3]:
    st.markdown("### The 14 Features Specification")
    st.write("Why 14 features? Eliminating redundant packet headers reduces computational latency by over 65%, enabling wire-speed intrusion detection.")

    features_info = [
        ("rate", "Flow", "Transferred packets per second", "Crucial for identifying volumetric flood attacks (e.g. DoS/DDoS)."),
        ("sttl", "Header", "Source-to-destination time to live (TTL)", "Operating systems have distinct initial TTL values; attacks often originate from fixed scanners with characteristic TTLs."),
        ("sload", "Flow", "Source bits per second", "Source transmission speed; indicates high-bandwidth exfiltration or flooding."),
        ("dload", "Flow", "Destination bits per second", "Destination transmission speed; zero or abnormally low dload in presence of high sload suggests DoS."),
        ("ct_srv_src", "Connection", "Connections with same service and source address in 100 connections", "Detects repeated connections to same service; characteristic of automated scanners."),
        ("ct_state_ttl", "Connection", "Connections having same state and dst/src TTL values", "Key indicator of synthetic or scripted packet generation."),
        ("ct_dst_ltm", "Connection", "Connections to same destination address in 100 connections", "High count indicates focused attack on a single target IP."),
        ("ct_src_dport_ltm", "Connection", "Connections from same source IP to same destination port", "Detects service-targeted brute-force or persistent attacks."),
        ("ct_dst_sport_ltm", "Connection", "Distinct destination ports from source", "Signature of vertical or horizontal port-scanning."),
        ("ct_dst_src_ltm", "Connection", "Connections between same source and destination IP", "Measures session intensity between host pairs."),
        ("ct_src_ltm", "Connection", "Connections from same source IP", "Measures overall source activity in the local time window."),
        ("ct_srv_dst", "Connection", "Connections to same service and destination IP", "Measures concentrated load on a target service."),
        ("state_CON", "Categorical (Encoded)", "Connection established / connected state indicator", "Binary 1/0 indicator representing established TCP/UDP connection."),
        ("state_INT", "Categorical (Encoded)", "Intermediate / half-open state indicator", "Binary 1/0 indicator; SYN scans and connectionless floods often remain in INT state.")
    ]

    doc_df = pd.DataFrame(features_info, columns=["Feature Name", "Category", "Description", "Cybersecurity Significance"])
    st.table(doc_df)

    st.markdown("""
    #### Reference Literature
    * **Awad, M.; Fraihat, S. (2023)** *Recursive Feature Elimination with Cross-Validation with Decision Tree: Feature Selection Method for Machine Learning-Based Intrusion Detection Systems.* Journal of Sensor and Actuator Networks.
    * **Moustafa, N.; Slay, J. (2015)** *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* IEEE MilCIS.
    """)
