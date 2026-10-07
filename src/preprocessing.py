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

CATEGORICAL_COLUMNS = ["instr", "class"]

NUMERIC_COLUMNS = [
    "nb.repeat", "attendance", "difficulty",
    "Q1", "Q2", "Q3", "Q4",
    "Q5", "Q6", "Q7", "Q8"
]


class StandardScaler:
    """Chuẩn hóa từng cột bằng thống kê học từ tập train."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    @staticmethod
    def _to_array(X):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("X phải là dữ liệu hai chiều.")
        return values

    def fit(self, X):
        """Học trung bình và độ lệch chuẩn của từng cột."""
        values = self._to_array(X)
        if len(values) == 0:
            raise ValueError("Không thể fit trên dữ liệu rỗng.")

        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0, ddof=0)

        # Tránh chia cho 0 khi một cột có giá trị không đổi.
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X):
        """Chuẩn hóa bằng thống kê đã học, không tính lại."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler chưa được fit.")

        values = self._to_array(X)
        if values.shape[1] != len(self.mean_):
            raise ValueError("Số cột không khớp với dữ liệu đã fit.")

        return (values - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit rồi chuẩn hóa dữ liệu."""
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Mã hóa category từ train và bỏ qua category chưa gặp."""

    def __init__(self):
        self.columns_ = None
        self.categories_ = None

    def fit(self, X):
        """Lưu tên cột và các category theo thứ tự ổn định."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X của OneHotEncoder phải là DataFrame.")
        if X.isna().any().any():
            raise ValueError("Các cột categorical có giá trị thiếu.")

        self.columns_ = X.columns.tolist()
        self.categories_ = {
            col: np.sort(X[col].unique())
            for col in self.columns_
        }
        return self

    def transform(self, X):
        """Trả về ma trận one-hot với cấu trúc đã học từ train."""
        if self.columns_ is None:
            raise ValueError("OneHotEncoder chưa được fit.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X của OneHotEncoder phải là DataFrame.")

        missing_cols = [
            col for col in self.columns_ if col not in X.columns
        ]
        if missing_cols:
            raise ValueError(f"Thiếu cột categorical: {missing_cols}")

        # Category lạ không khớp category nào nên cả nhóm bằng 0.
        encoded_columns = [
            (X[col].to_numpy() == category).astype(float)
            for col in self.columns_
            for category in self.categories_[col]
        ]

        if not encoded_columns:
            return np.empty((len(X), 0), dtype=float)

        return np.column_stack(encoded_columns)

    def fit_transform(self, X):
        """Fit rồi mã hóa dữ liệu."""
        return self.fit(X).transform(X)

    def get_feature_names_out(self):
        """Trả về tên feature theo thứ tự của ma trận output."""
        if self.columns_ is None:
            raise ValueError("OneHotEncoder chưa được fit.")

        return [
            f"{col}_{category}"
            for col in self.columns_
            for category in self.categories_[col]
        ]


class DataPreprocessor:
    """Ghép one-hot categorical và numeric đã chuẩn hóa."""

    def __init__(self, categorical_cols, numeric_cols):
        self.categorical_cols = list(categorical_cols)
        self.numeric_cols = list(numeric_cols)
        self.encoder = OneHotEncoder()
        self.scaler = StandardScaler()

    def fit(self, X):
        """Học category và thống kê numeric từ dữ liệu đầu vào."""
        self.encoder.fit(X[self.categorical_cols])
        self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        """Trả về ma trận float: categorical trước, numeric sau."""
        categorical = self.encoder.transform(X[self.categorical_cols])
        numeric = self.scaler.transform(X[self.numeric_cols])

        return np.concatenate(
            [categorical, numeric], axis=1
        ).astype(float)

    def fit_transform(self, X):
        """Fit rồi biến đổi dữ liệu."""
        return self.fit(X).transform(X)

    def get_feature_names_out(self):
        """Trả về tên feature theo đúng thứ tự output."""
        return self.encoder.get_feature_names_out() + self.numeric_cols
