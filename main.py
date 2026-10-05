import sys

from src.data_loader import LABEL_NAMES, load_dataset
from src.preprocessing import train_test_split_stratified


def print_distribution(name, labels):
    """In số lượng và tỷ lệ phần trăm của từng lớp."""
    counts = labels.value_counts().reindex(list(LABEL_NAMES), fill_value=0)
    percentages = counts / len(labels) * 100
    print(f"{name} - số lượng:")
    print(counts.to_string())
    print(f"{name} - tỷ lệ (%):")
    print(percentages.round(2).to_string())


def main():
    
    # Bảo đảm console Windows in được tên nhãn tiếng Việt.
    sys.stdout.reconfigure(encoding="utf-8")
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=0.2, random_state=42
    )

    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    print("Features:", X.columns.tolist())
    print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape)
    print("LABEL_NAMES:", LABEL_NAMES)
    print_distribution("Toàn bộ", y)
    print_distribution("Train", y_train)
    print_distribution("Test", y_test)


if __name__ == "__main__":
    main()
