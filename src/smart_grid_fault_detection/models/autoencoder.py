from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from rich.console import Console
from sklearn.preprocessing import StandardScaler, MinMaxScaler

console = Console()


@dataclass
class AutoencoderConfig:
    input_path: Path = Path("data/processed/smart_grid_clean.parquet")
    timestamp_col: str = "timestamp"
    feature_cols: Optional[List[str]] = None
    drop_columns: List[str] = field(default_factory=lambda: ["fault_flag", "fault_type"])
    sequence_length: int = 32
    batch_size: int = 64
    epochs: int = 40
    latent_dim: int = 32
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    val_fraction: float = 0.2
    scaler: str = "standard"  # standard or minmax
    learning_rate: float = 1e-3
    patience: int = 5
    output_dir: Path = Path("models/autoencoder")
    random_state: int = 13


def _load_dataset(cfg: AutoencoderConfig) -> pd.DataFrame:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.input_path}")
    if cfg.input_path.suffix == ".parquet":
        df = pd.read_parquet(cfg.input_path)
    else:
        df = pd.read_csv(cfg.input_path, parse_dates=[cfg.timestamp_col])
    df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)
    for col in cfg.drop_columns:
        if col in df:
            df = df.drop(columns=col)
    return df


def _select_features(df: pd.DataFrame, cfg: AutoencoderConfig) -> List[str]:
    if cfg.feature_cols:
        missing = [c for c in cfg.feature_cols if c not in df]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        return cfg.feature_cols
    return df.select_dtypes(include=["number"]).columns.tolist()


def _build_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler '{name}'")


def _create_sequences(values: np.ndarray, timestamps: pd.Series, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    ts = []
    for start in range(0, len(values) - seq_len + 1):
        end = start + seq_len
        sequences.append(values[start:end])
        ts.append(timestamps.iloc[end - 1])
    if not sequences:
        raise ValueError("Not enough samples to create sequences. Reduce sequence_length.")
    return np.stack(sequences), np.array(ts)


def _build_model(cfg: AutoencoderConfig, n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(cfg.sequence_length, n_features))
    x = inputs
    for units in cfg.hidden_units:
        x = tf.keras.layers.LSTM(units, return_sequences=True, dropout=cfg.dropout)(x)
    x = tf.keras.layers.LSTM(cfg.latent_dim, return_sequences=False, dropout=cfg.dropout)(x)
    x = tf.keras.layers.RepeatVector(cfg.sequence_length)(x)
    for units in reversed(cfg.hidden_units):
        x = tf.keras.layers.LSTM(units, return_sequences=True, dropout=cfg.dropout)(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss="mse")
    return model


def run_autoencoder(cfg: AutoencoderConfig) -> dict[str, Path]:
    tf.keras.utils.set_random_seed(cfg.random_state)

    df = _load_dataset(cfg)
    feature_cols = _select_features(df, cfg)
    values = df[feature_cols].to_numpy(dtype=np.float32)

    scaler = _build_scaler(cfg.scaler)
    values_scaled = scaler.fit_transform(values)

    sequences, seq_timestamps = _create_sequences(values_scaled, df[cfg.timestamp_col], cfg.sequence_length)
    n_samples = len(sequences)
    val_len = int(n_samples * cfg.val_fraction)
    if val_len == 0:
        X_train = sequences
        X_val = None
    else:
        X_train = sequences[:-val_len]
        X_val = sequences[-val_len:]

    model = _build_model(cfg, len(feature_cols))

    callbacks: List[tf.keras.callbacks.Callback] = []
    if cfg.patience > 0 and val_len > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.patience, restore_best_weights=True
            )
        )

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val) if X_val is not None else None,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    recon = model.predict(sequences, batch_size=cfg.batch_size, verbose=0)
    errors = np.mean(np.mean(np.square(sequences - recon), axis=2), axis=1)
    train_errors = errors[: len(X_train)]
    threshold = float(np.percentile(train_errors, 95))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.output_dir / "autoencoder.keras"
    model.save(model_path, include_optimizer=True)

    scaler_path = cfg.output_dir / "scaler.joblib"
    joblib.dump({"scaler": scaler, "features": feature_cols}, scaler_path)

    history_path = cfg.output_dir / "history.json"
    history_path.write_text(json.dumps(history.history, indent=2))

    recon_df = pd.DataFrame(
        {
            cfg.timestamp_col: seq_timestamps,
            "reconstruction_error": errors,
        }
    )
    recon_df["threshold_95"] = threshold
    errors_path = cfg.output_dir / "reconstruction_errors.csv"
    recon_df.to_csv(errors_path, index=False)

    metadata = {
        "sequence_length": cfg.sequence_length,
        "feature_count": len(feature_cols),
        "train_samples": len(X_train),
        "val_samples": int(X_val.shape[0]) if X_val is not None else 0,
        "threshold_95": threshold,
        "scaler": cfg.scaler,
    }
    metadata_path = cfg.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    console.print(f"[green]Saved LSTM autoencoder model:[/] {model_path}")
    console.print(f"[green]Saved reconstruction errors:[/] {errors_path}")

    return {
        "model": model_path,
        "scaler": scaler_path,
        "history": history_path,
        "errors": errors_path,
        "metadata": metadata_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an LSTM autoencoder for smart-grid anomaly detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=AutoencoderConfig.input_path)
    parser.add_argument("--sequence-length", type=int, default=AutoencoderConfig.sequence_length)
    parser.add_argument("--batch-size", type=int, default=AutoencoderConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=AutoencoderConfig.epochs)
    parser.add_argument("--latent-dim", type=int, default=AutoencoderConfig.latent_dim)
    parser.add_argument("--hidden-units", type=int, nargs="*", default=None)
    parser.add_argument("--dropout", type=float, default=AutoencoderConfig.dropout)
    parser.add_argument("--val-fraction", type=float, default=AutoencoderConfig.val_fraction)
    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--learning-rate", type=float, default=AutoencoderConfig.learning_rate)
    parser.add_argument("--patience", type=int, default=AutoencoderConfig.patience)
    parser.add_argument("--output-dir", type=Path, default=AutoencoderConfig.output_dir)
    parser.add_argument("--features", type=str, nargs="*", help="Optional feature subset")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = AutoencoderConfig(
        input_path=args.input,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        hidden_units=args.hidden_units or AutoencoderConfig().hidden_units,
        dropout=args.dropout,
        val_fraction=args.val_fraction,
        scaler=args.scaler,
        learning_rate=args.learning_rate,
        patience=args.patience,
        output_dir=args.output_dir,
        feature_cols=args.features or None,
    )

    run_autoencoder(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
