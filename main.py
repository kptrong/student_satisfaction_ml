import sys

from src.data_loader import LABEL_NAMES, load_dataset


def main():
    # Bảo đảm console Windows in được tên nhãn tiếng Việt.
    sys.stdout.reconfigure(encoding="utf-8")
    X, y = load_dataset()
    counts = y.value_counts().reindex(list(LABEL_NAMES), fill_value=0)
    percentages = counts / len(y) * 100

    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    print("Features:", X.columns.tolist())
    print("Số lượng từng nhãn:")
    print(counts.to_string())
    print("Tỷ lệ từng nhãn (%):")
    print(percentages.round(2).to_string())
    print("LABEL_NAMES:", LABEL_NAMES)


if __name__ == "__main__":
    main()
