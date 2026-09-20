import numpy as np
import pandas as pd

from aml_sequence_pipeline import (
    build_sequence_tensors,
    positive_class_weight,
    stratified_account_split,
    window_labels,
)


def sample_data():
    return pd.DataFrame(
        {
            "nameDest": ["A", "B", "B", "B", "B", "B"],
            "step": [1, 1, 2, 3, 4, 5],
            "value": [10, 1, 2, 3, 4, 5],
            "isFraud": [0, 1, 0, 0, 0, 0],
        }
    )


def test_padding_is_on_the_right_and_recent_values_are_kept():
    tensors = build_sequence_tensors(
        sample_data(), ["A", "B"], ["value"], max_len=3
    )
    by_account = {account: i for i, account in enumerate(tensors.account_ids)}

    a = by_account["A"]
    b = by_account["B"]

    np.testing.assert_array_equal(tensors.X[a, :, 0], [10, 0, 0])
    np.testing.assert_array_equal(tensors.mask[a], [1, 0, 0])
    np.testing.assert_array_equal(tensors.X[b, :, 0], [3, 4, 5])
    np.testing.assert_array_equal(tensors.mask[b], [1, 1, 1])


def test_label_is_computed_from_the_visible_window():
    tensors = build_sequence_tensors(
        sample_data(), ["A", "B"], ["value"], max_len=3
    )
    b = list(tensors.account_ids).index("B")

    assert tensors.history_y[b] == 1
    assert tensors.y[b] == 0

    labels = window_labels(sample_data(), max_len=3)
    assert labels.loc["B", "history_y"] == 1
    assert labels.loc["B", "window_y"] == 0


def test_account_split_is_disjoint_and_stratified():
    labels = pd.Series(
        [0] * 80 + [1] * 20,
        index=[f"account_{i}" for i in range(100)],
        dtype=np.int8,
    )
    train, val, test = stratified_account_split(labels)

    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert labels.loc[train].mean() == 0.2
    assert np.isclose(labels.loc[val].mean(), 0.2, atol=0.04)
    assert np.isclose(labels.loc[test].mean(), 0.2, atol=0.04)


def test_positive_weight_uses_natural_training_distribution():
    assert positive_class_weight([0, 0, 0, 1]) == 3.0
