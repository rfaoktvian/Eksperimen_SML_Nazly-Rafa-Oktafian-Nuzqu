# automate_Nazly-Rafa-Oktafian-Nuzqu.py
# Skrip otomatisasi preprocessing - Heart Disease Dataset
# Kriteria 1 Level Skilled
# Autor: Nazly Rafa Oktafian Nuzqu

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os
import argparse


def load_data(filepath: str) -> pd.DataFrame:
    print(f"[INFO] Memuat dataset dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Shape dataset: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[INFO] Duplikat dihapus: {before - len(df)} | Sisa: {len(df)}")
    return df


def remove_outliers(X, y, contamination=0.05):
    iso = IsolationForest(contamination=contamination, random_state=42)
    mask = iso.fit_predict(X) == 1
    print(f"[INFO] Outlier dihapus: {(~mask).sum()} | Sisa: {mask.sum()}")
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    print("[INFO] Standarisasi selesai.")
    return X_train_s, X_test_s, scaler


def save_preprocessed(X_train, X_test, y_train, y_test, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'heart_train_preprocessed.csv')
    test_path  = os.path.join(output_dir, 'heart_test_preprocessed.csv')

    train_df = X_train.copy(); train_df['target'] = y_train.values
    test_df  = X_test.copy();  test_df['target']  = y_test.values

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    print(f"[INFO] Disimpan: {train_path}")
    print(f"[INFO] Disimpan: {test_path}")


def run_pipeline(input_file, output_dir='preprocessed_data',
                 contamination=0.05, test_size=0.2):
    print("=" * 55)
    print("   PIPELINE PREPROCESSING - Heart Disease Dataset")
    print("   Autor: Nazly Rafa Oktafian Nuzqu")
    print("=" * 55)
    df = load_data(input_file)
    df = remove_duplicates(df)
    X  = df.drop(columns=['target'])
    y  = df['target']
    X, y = remove_outliers(X, y, contamination)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    X_train_s, X_test_s, _ = scale_features(X_train, X_test)
    save_preprocessed(X_train_s, X_test_s, y_train, y_test, output_dir)
    print("\n[INFO] Pipeline selesai!")
    print("=" * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',         type=str,   default='../heart_raw.csv')
    parser.add_argument('--output',        type=str,   default='heart_preprocessing')
    parser.add_argument('--contamination', type=float, default=0.05)
    parser.add_argument('--test_size',     type=float, default=0.2)
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.contamination, args.test_size)
