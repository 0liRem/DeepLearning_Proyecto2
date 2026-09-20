"""Funciones de preparación de secuencias para el proyecto AML."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SequenceTensors:
    X: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    account_ids: np.ndarray
    lengths: np.ndarray
    history_y: np.ndarray


def positive_class_weight(y) -> float:
    """Peso positivo para BCE usando la distribución natural de train.

    Este peso se usa con un DataLoader que solo baraja los ejemplos; no debe
    combinarse con un sampler que vuelva a balancear las clases.
    """
    values = np.asarray(y)
    positives = float(values.sum())
    negatives = float(len(values) - positives)
    if positives == 0:
        raise ValueError("El conjunto de entrenamiento no contiene positivos.")
    return negatives / positives


def stratified_account_split(
    labels: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separa cuentas completas manteniendo la proporción de clases."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Las proporciones de train, validación y prueba deben sumar 1.")

    accounts = labels.index.to_numpy()
    y = labels.to_numpy()
    train_accounts, other_accounts, _, other_y = train_test_split(
        accounts,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val = val_size / (val_size + test_size)
    val_accounts, test_accounts = train_test_split(
        other_accounts,
        train_size=relative_val,
        random_state=random_state,
        stratify=other_y,
    )

    if set(train_accounts) & set(val_accounts):
        raise AssertionError("Hay cuentas repetidas entre train y validación.")
    if set(train_accounts) & set(test_accounts):
        raise AssertionError("Hay cuentas repetidas entre train y prueba.")
    if set(val_accounts) & set(test_accounts):
        raise AssertionError("Hay cuentas repetidas entre validación y prueba.")

    return train_accounts, val_accounts, test_accounts


def window_labels(
    df: pd.DataFrame,
    max_len: int,
    account_col: str = "nameDest",
    label_col: str = "isFraud",
    time_col: str = "step",
) -> pd.DataFrame:
    """Resume el label de la historia completa y de la ventana más reciente."""
    ordered = df.sort_values([account_col, time_col], kind="mergesort")
    recent = ordered.groupby(account_col, sort=False).tail(max_len)

    history = ordered.groupby(account_col, sort=False)[label_col].max()
    current = recent.groupby(account_col, sort=False)[label_col].max()
    lengths = recent.groupby(account_col, sort=False).size()

    return pd.DataFrame(
        {
            "history_y": history,
            "window_y": current.reindex(history.index, fill_value=0),
            "window_length": lengths.reindex(history.index, fill_value=0),
        }
    )


def build_sequence_tensors(
    df: pd.DataFrame,
    account_ids,
    feature_cols: list[str],
    max_len: int,
    account_col: str = "nameDest",
    label_col: str = "isFraud",
    time_col: str = "step",
) -> SequenceTensors:
    """Construye ventanas recientes con datos válidos a la izquierda.

    El padding queda al final. Esta disposición es necesaria para que
    ``pack_padded_sequence`` procese las transacciones reales de cada cuenta.
    El label corresponde solo a la ventana que entra al modelo.
    """
    account_ids = np.asarray(account_ids)
    selected = df[df[account_col].isin(account_ids)].copy()
    selected = selected.sort_values([account_col, time_col], kind="mergesort")

    if selected.empty:
        raise ValueError("No se encontraron transacciones para las cuentas solicitadas.")

    history_labels = selected.groupby(account_col, sort=False)[label_col].max()
    recent = selected.groupby(account_col, sort=False).tail(max_len).copy()

    # factorize conserva el orden de aparición después del ordenamiento estable.
    recent["_account_idx"], ordered_accounts = pd.factorize(
        recent[account_col], sort=False
    )
    recent["_position"] = recent.groupby("_account_idx", sort=False).cumcount()

    n_accounts = len(ordered_accounts)
    n_features = len(feature_cols)
    rows = recent["_account_idx"].to_numpy()
    cols = recent["_position"].to_numpy()

    X = np.zeros((n_accounts, max_len, n_features), dtype=np.float32)
    mask = np.zeros((n_accounts, max_len), dtype=np.float32)
    X[rows, cols] = recent[feature_cols].to_numpy(dtype=np.float32)
    mask[rows, cols] = 1.0

    window_y = (
        recent.groupby("_account_idx", sort=False)[label_col]
        .max()
        .reindex(range(n_accounts), fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    lengths = (
        recent.groupby("_account_idx", sort=False)
        .size()
        .reindex(range(n_accounts), fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    history_y = history_labels.reindex(ordered_accounts).to_numpy(dtype=np.float32)

    if not np.array_equal(mask.sum(axis=1).astype(np.int64), lengths):
        raise AssertionError("La máscara y las longitudes no coinciden.")
    if np.any(mask[np.arange(n_accounts), lengths - 1] != 1):
        raise AssertionError("La última posición válida no está marcada.")
    short_rows = np.flatnonzero(lengths < max_len)
    if np.any(mask[short_rows, lengths[short_rows]] != 0):
        raise AssertionError("Se encontró información después del final de una secuencia.")

    return SequenceTensors(
        X=X,
        mask=mask,
        y=window_y,
        account_ids=np.asarray(ordered_accounts),
        lengths=lengths,
        history_y=history_y,
    )
