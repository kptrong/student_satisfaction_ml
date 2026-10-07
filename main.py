import sys
import pandas as pd
from src.data_loader import LABEL_NAMES, load_dataset
from src.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    DataPreprocessor,
    train_test_split_stratified
)

def print_distribution(name, labels):
    """In số lượng và tỷ lệ phần trăm của từng lớp."""
    counts = labels.value_counts().reindex(list(LABEL_NAMES), fill_value=0)
    percentages = counts / len(labels) * 100
    print(f"{name} - số lượng:")
    print(counts.to_string())
    print(f"{name} - tỷ lệ (%):")
    print(percentages.round(2).to_string())


def main():
    """Đọc dữ liệu, chia train/test và kiểm tra preprocessing."""
    sys.stdout.reconfigure(encoding="utf-8")

    # Đọc dữ liệu và chia train/test theo lớp
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    print("Features gốc:", X.columns.tolist())
    print("LABEL_NAMES:", LABEL_NAMES)

    print("\nKích thước sau chia dữ liệu:")
    print("X_train.shape:", X_train.shape)
    print("X_test.shape: ", X_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape: ", y_test.shape)

    # Chỉ fit preprocessing trên train; test chỉ được transform
    preprocessor = DataPreprocessor(
        categorical_cols=CATEGORICAL_COLUMNS,
        numeric_cols=NUMERIC_COLUMNS
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nKích thước sau preprocessing:")
    print("X_train_processed.shape:", X_train_processed.shape)
    print("X_test_processed.shape: ", X_test_processed.shape)
    print(
        "Features sau preprocessing:",
        preprocessor.get_feature_names_out()
    )

    # Numeric nằm sau toàn bộ phần one-hot trong ma trận output
    n_onehot = len(preprocessor.encoder.get_feature_names_out())
    train_numeric = X_train_processed[:, n_onehot:]

    numeric_stats = pd.DataFrame({
        "Mean": train_numeric.mean(axis=0),
        "Std": train_numeric.std(axis=0, ddof=0)
    }, index=NUMERIC_COLUMNS)

    print("\nThống kê numeric trên TRAIN sau chuẩn hóa:")
    print(numeric_stats.round(6).to_string())


if __name__ == "__main__":
    main()
