from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from rich.console import Console
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt

console = Console()


@dataclass
class CNNConfig:
    input_path: Path = Path("data/processed/smart_grid_clean.parquet")
    timestamp_col: str = "timestamp"
    label_col: str = "fault_type"
    drop_columns: List[str] = field(default_factory=lambda: ["fault_flag"])
    sequence_length: int = 48
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 1e-3
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 13
    output_dir: Path = Path("models/cnn_classifier")


def _load_dataset(cfg: CNNConfig) -> pd.DataFrame:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.input_path}")
    df = pd.read_parquet(cfg.input_path)
    df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)
    for col in cfg.drop_columns:
        if col in df:
            df = df.drop(columns=col)
    return df


def _build_sequences(df: pd.DataFrame, cfg: CNNConfig):
    feature_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if cfg.label_col in feature_cols:
        feature_cols.remove(cfg.label_col)
    label_series = df[cfg.label_col]
    timestamps = df[cfg.timestamp_col]
    values = df[feature_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    values = scaler.fit_transform(values)

    sequences = []
    labels = []
    seq_timestamps = []
    for start in range(0, len(values) - cfg.sequence_length + 1):
        end = start + cfg.sequence_length
        sequences.append(values[start:end])
        labels.append(label_series.iloc[end - 1])
        seq_timestamps.append(timestamps.iloc[end - 1])
    sequences = np.stack(sequences)
    label_arr = np.array(labels)
    unique_labels = sorted(pd.unique(label_arr))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in label_arr])

    seq_timestamps = np.array(seq_timestamps)
    return sequences, y, seq_timestamps, feature_cols, scaler, label_map


def _train_val_test_split(sequences, labels, timestamps, cfg: CNNConfig):
    temp_size = cfg.val_fraction + cfg.test_fraction
    X_train, X_temp, y_train, y_temp, ts_train, ts_temp = train_test_split(
        sequences,
        labels,
        timestamps,
        test_size=temp_size,
        stratify=labels,
        shuffle=True,
        random_state=cfg.random_state,
    )
    if temp_size == 0:
        return (X_train, y_train, ts_train), (np.empty((0,)), np.empty((0,)), np.empty((0,))), (
            np.empty((0,)),
            np.empty((0,)),
            np.empty((0,)),
        )

    val_ratio = cfg.val_fraction / temp_size
    X_val, X_test, y_val, y_test, ts_val, ts_test = train_test_split(
        X_temp,
        y_temp,
        ts_temp,
        test_size=cfg.test_fraction / temp_size,
        stratify=y_temp,
        shuffle=True,
        random_state=cfg.random_state,
    )
    return (X_train, y_train, ts_train), (X_val, y_val, ts_val), (X_test, y_test, ts_test)


def _build_model(cfg: CNNConfig, n_features: int, n_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(cfg.sequence_length, n_features))
    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_cnn(cfg: CNNConfig) -> Dict[str, Path]:
    tf.keras.utils.set_random_seed(cfg.random_state)

    df = _load_dataset(cfg)
    sequences, labels, seq_timestamps, feature_cols, scaler, label_map = _build_sequences(df, cfg)
    (X_train, y_train, ts_train), (X_val, y_val, ts_val), (X_test, y_test, ts_test) = _train_val_test_split(
        sequences, labels, seq_timestamps, cfg
    )

    model = _build_model(cfg, sequences.shape[2], len(label_map))
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weight = {cls: weight for cls, weight in zip(classes, weights)}

    callbacks = []
    if len(X_val) > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    idx_to_label = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(idx_to_label))),
        target_names=idx_to_label,
        output_dict=True,
        zero_division=0,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.output_dir / "cnn_fault_classifier.keras"
    model.save(model_path)

    scaler_path = cfg.output_dir / "scaler.joblib"
    joblib.dump({"scaler": scaler, "features": feature_cols}, scaler_path)

    label_map_path = cfg.output_dir / "label_map.json"
    label_map_path.write_text(json.dumps(label_map, indent=2))

    history_path = cfg.output_dir / "history.json"
    history_path.write_text(json.dumps(history.history, indent=2))

    report_path = cfg.output_dir / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(idx_to_label))))
    cm_path = cfg.output_dir / "confusion_matrix.json"
    cm_path.write_text(json.dumps({"labels": idx_to_label, "matrix": cm.tolist()}, indent=2))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=idx_to_label, yticklabels=idx_to_label)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CNN Fault-Type Confusion Matrix")
    plt.tight_layout()
    cm_plot = cfg.output_dir / "confusion_matrix.png"
    plt.savefig(cm_plot, dpi=200)
    plt.close()

    preds_df = pd.DataFrame(
        {
            cfg.timestamp_col: ts_test,
            "true_label": [idx_to_label[idx] for idx in y_test],
            "pred_label": [idx_to_label[idx] for idx in y_pred],
        }
    )
    for idx, label in enumerate(idx_to_label):
        preds_df[f"prob_{label}"] = preds[:, idx]
    preds_path = cfg.output_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    metrics_summary = {
        "test_loss": float(test_metrics[0]),
        "test_accuracy": float(test_metrics[1]),
    }
    metrics_path = cfg.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2))

    console.print(f"[green]Saved CNN classifier:[/] {model_path}")
    console.print(f"[green]Test accuracy:[/] {metrics_summary['test_accuracy']:.3f}")

    return {
        "model": model_path,
        "scaler": scaler_path,
        "label_map": label_map_path,
        "report": report_path,
        "metrics": metrics_path,
        "confusion_matrix": cm_path,
        "confusion_matrix_plot": cm_plot,
        "predictions": preds_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 1D CNN to classify spike/dropout/cyber anomalies from smart-grid sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=CNNConfig.input_path)
    parser.add_argument("--sequence-length", type=int, default=CNNConfig.sequence_length)
    parser.add_argument("--epochs", type=int, default=CNNConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=CNNConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=CNNConfig.learning_rate)
    parser.add_argument("--output-dir", type=Path, default=CNNConfig.output_dir)
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = CNNConfig(
        input_path=args.input,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    run_cnn(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
