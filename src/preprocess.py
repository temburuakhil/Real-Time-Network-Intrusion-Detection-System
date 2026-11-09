"""
Network Intrusion Detection System (NIDS) - Data Preprocessing Module
Extracts the exact 14 features from UNSW-NB15 dataset, encodes categorical
state columns, and applies MinMaxScaler [0, 1].
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Exact 14 features specified
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

CATEGORICAL_FEATURES = [
    'state_CON',
    'state_INT'
]

FEATURE_NAMES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'label'


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the 14 features and handles state one-hot encoding.
    """
    df_copy = df.copy()
    
    # One-hot encode state column for CON and INT
    if 'state' in df_copy.columns:
        df_copy['state_CON'] = (df_copy['state'] == 'CON').astype(float)
        df_copy['state_INT'] = (df_copy['state'] == 'INT').astype(float)
    else:
        if 'state_CON' not in df_copy.columns:
            df_copy['state_CON'] = 0.0
        if 'state_INT' not in df_copy.columns:
            df_copy['state_INT'] = 0.0

    return df_copy[FEATURE_NAMES]


def load_and_preprocess(
    train_path: str = 'Training and Testing Sets/UNSW_NB15_training-set.csv',
    test_path: str = 'Training and Testing Sets/UNSW_NB15_testing-set.csv',
    scaler_save_path: str = 'models/scaler.joblib'
):
    """
    Loads training and testing CSVs, filters to 14 features + target,
    fits MinMaxScaler on training set and transforms both sets.
    Returns:
        X_train_scaled, y_train, X_test_scaled, y_test, scaler
    """
    print(f"Loading training data from: {train_path}")
    train_raw = pd.read_csv(train_path)
    print(f"Loading testing data from: {test_path}")
    test_raw = pd.read_csv(test_path)

    # Extract target
    y_train = train_raw[TARGET_COLUMN].values
    y_test = test_raw[TARGET_COLUMN].values

    # Extract features
    X_train_raw = extract_features(train_raw)
    X_test_raw = extract_features(test_raw)

    print(f"Extracted features shape: Train {X_train_raw.shape}, Test {X_test_raw.shape}")
    print(f"Features: {FEATURE_NAMES}")

    # Scale with MinMaxScaler fit on training set
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=FEATURE_NAMES
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=FEATURE_NAMES
    )

    # Save scaler for inference / deployment
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"MinMaxScaler saved to: {scaler_save_path}")

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess()
    print("Preprocessing completed successfully.")
    print("Train features sample:")
    print(X_train.head(3))
