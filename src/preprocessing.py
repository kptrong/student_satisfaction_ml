"""Các thao tác chuẩn bị dữ liệu không phụ thuộc scikit-learn."""

import numpy as np
import pandas as pd


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """Chia train/test theo tỷ lệ từng lớp và giữ nguyên kiểu pandas."""
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X phải là DataFrame và y phải là Series.")
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số dòng.")
    if not 0 < test_size < 1:
        raise ValueError("test_size phải nằm trong khoảng (0, 1).")
    if y.isna().any():
        raise ValueError("y không được chứa giá trị thiếu.")

    rng = np.random.default_rng(random_state)
    y_values = y.to_numpy()
    train_parts = []
    test_parts = []

    for label in np.unique(y_values):
        # Lấy vị trí theo lớp, không phụ thuộc index hiện tại của Series.
        class_indices = np.flatnonzero(y_values == label)
        if len(class_indices) < 2:
            raise ValueError(f"Lớp {label!r} cần ít nhất 2 mẫu để chia train/test.")

        rng.shuffle(class_indices)
        n_test = int(round(len(class_indices) * test_size))
        n_test = min(max(n_test, 1), len(class_indices) - 1)
        test_parts.append(class_indices[:n_test])
        train_parts.append(class_indices[n_test:])

    # Trộn lần cuối để các lớp không nằm thành từng khối liên tiếp.
    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    return X_train, X_test, y_train, y_test
