"""
Network Intrusion Detection System (NIDS) - Model Training and Evaluation
Trains 6 ML models on the 14-feature UNSW-NB15 dataset:
1. Logistic Regression
2. Naive Bayes (GaussianNB)
3. Stochastic Gradient Descent (SGD)
4. Random Forest
5. AdaBoost
6. Multi-Layer Perceptron (MLP)

Evaluates on the testing set and produces metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Training Time
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from preprocess import load_and_preprocess, FEATURE_NAMES

RANDOM_STATE = 42

def get_models():
    """
    Returns a dictionary of the 6 models configured with random_state=42.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver='lbfgs'
        ),
        "Naive Bayes": GaussianNB(),
        "Stochastic Gradient Descent": SGDClassifier(
            loss='log_loss',
            random_state=RANDOM_STATE,
            max_iter=1000,
            tol=1e-3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=RANDOM_STATE,
            n_estimators=50
        ),
        "Multi-Layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            batch_size=512,
            max_iter=50,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=RANDOM_STATE
        )
    }
    return models


def train_and_evaluate_all():
    # 1. Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess()

    models = get_models()
    results = []
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*60, flush=True)
    print("STARTING TRAINING OF 6 MODELS ON 14 FEATURES", flush=True)
    print("="*60, flush=True)

    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        filename_safe = name.lower().replace(" ", "_").replace("-", "_")
        model_path = f"models/{filename_safe}.joblib"
        
        print(f"\n>>> Processing {name}...", flush=True)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"    Completed training in {train_time:.2f} seconds.", flush=True)

        # Save model with compression (keeps files under GitHub 100MB limit)
        joblib.dump(model, model_path, compress=3)
        print(f"    Model saved to {model_path}", flush=True)

        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC / AUC
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred
            auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
        except Exception:
            auc = np.nan

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4) if not np.isnan(auc) else None,
            "Training Time (s)": round(train_time, 2),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        })

    # Save ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - 14-Feature NIDS Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_plot_path = "results/roc_curves.png"
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nROC curves plot saved to: {roc_plot_path}", flush=True)

    # Build and print results DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "="*80, flush=True)
    print("FINAL EVALUATION RESULTS TABLE (14 FEATURES)", flush=True)
    print("="*80, flush=True)
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Training Time (s)"]
    print(results_df[display_cols].to_string(index=False), flush=True)

    # Save CSV and JSON
    results_df.to_csv("results/model_comparison.csv", index=False)
    with open("results/model_comparison.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults exported to results/model_comparison.csv and results/model_comparison.json", flush=True)

    # Plot Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_plot = results_df.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Performance Metrics Comparison Across 6 Models (14 Features)")
    plt.ylim(0.6, 1.0)
    plt.xticks(rotation=20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    metrics_plot_path = "results/model_comparison_chart.png"
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()
    print(f"Metrics comparison chart saved to: {metrics_plot_path}", flush=True)

    return results_df


if __name__ == '__main__':
    train_and_evaluate_all()
