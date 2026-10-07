import sys
import pandas as pd
from src.data_loader import LABEL_NAMES, load_dataset
from src.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    DataPreprocessor,
    train_test_split_stratified
)
import numpy as np
import pandas as pd
from src.metrics import evaluate_classification

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

    # Chỉ fit preprocessing trên train; test chỉ được transform
    preprocessor = DataPreprocessor(
        categorical_cols=CATEGORICAL_COLUMNS,
        numeric_cols=NUMERIC_COLUMNS
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)


    # Numeric nằm sau toàn bộ phần one-hot trong ma trận output
    n_onehot = len(preprocessor.encoder.get_feature_names_out())
    train_numeric = X_train_processed[:, n_onehot:]

    numeric_stats = pd.DataFrame({
        "Mean": train_numeric.mean(axis=0),
        "Std": train_numeric.std(axis=0, ddof=0)
    }, index=NUMERIC_COLUMNS)
    
    # Kiểm tra metrics bằng dữ liệu demo, chưa dùng model thật.
    y_true_demo = np.array([0, 0, 1, 1, 2, 2])
    y_pred_demo = np.array([0, 1, 1, 1, 2, 0])

    demo_result = evaluate_classification(
        y_true_demo,
        y_pred_demo,
        labels=[0, 1, 2]
    )

    print("\nKiểm tra classification metrics:")
    print("Thứ tự nhãn:", demo_result["labels"])
    print("Confusion Matrix:")
    print(demo_result["confusion_matrix"])

    print(f"Accuracy:        {demo_result['accuracy']:.4f}")
    print(f"Precision macro: {demo_result['precision_macro']:.4f}")
    print(f"Recall macro:    {demo_result['recall_macro']:.4f}")
    print(f"F1 macro:        {demo_result['f1_macro']:.4f}")

    per_class_df = pd.DataFrame.from_dict(
        demo_result["per_class"],
        orient="index"
    )
    per_class_df.index.name = "Class"

    print("\nMetric từng class:")
    print(per_class_df.round(4).to_string())

if __name__ == "__main__":
    main()
