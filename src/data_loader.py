"""Đọc dữ liệu và tạo đầu vào cho bài toán dự đoán mức hài lòng."""

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "instr",
    "class",
    "nb.repeat",
    "attendance",
    "difficulty",
    "Q1", "Q2", "Q3", "Q4",
    "Q5", "Q6", "Q7", "Q8",
]

TARGET_COLUMN = "Q10"

LABEL_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
LABEL_NAMES = {0: "Thấp", 1: "Trung lập", 2: "Cao"}


def load_raw_data(csv_path=None):
    """Đọc CSV gốc; mặc định tìm trong thư mục data của project."""
    if csv_path is None:
        csv_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "turkiye-student-evaluation_generic.csv"
        )

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path.resolve()}")

    return pd.read_csv(csv_path)


def load_dataset(csv_path=None):
    """Trả về X và y ba lớp, giữ nguyên các dòng trùng hoàn toàn."""
    raw_data = load_raw_data(csv_path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    if missing_columns:
        raise ValueError(f"Thiếu các cột cần thiết: {missing_columns}")

    data = raw_data[required_columns].copy()
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data[data[TARGET_COLUMN].isin(LABEL_MAP)].copy()
    missing_features = data[FEATURE_COLUMNS].columns[
        data[FEATURE_COLUMNS].isna().any()
    ].tolist()
    if missing_features:
        raise ValueError(f"Feature có giá trị thiếu hoặc không phải số: {missing_features}")

    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].map(LABEL_MAP).astype(int)
    return X, y
