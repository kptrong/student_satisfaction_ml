"""Classification metrics tự cài bằng NumPy."""

import numpy as np


def _validate_inputs(y_true, y_pred):
    """Chuyển đầu vào thành array 1 chiều và kiểm tra kích thước."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.size != y_pred.size:
        raise ValueError("y_true và y_pred phải có cùng số phần tử.")
    if y_true.size == 0:
        raise ValueError("Đầu vào không được rỗng.")

    return y_true, y_pred


def confusion_matrix(y_true, y_pred, labels=None):
    """Đếm Actual theo hàng, Predicted theo cột; trả về cm và labels."""
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    if labels is None:
        labels = np.unique(
            np.concatenate([y_true, y_pred])
        ).tolist()
    else:
        labels = np.asarray(labels).reshape(-1).tolist()

    if not labels:
        raise ValueError("labels không được rỗng.")
    if len(set(labels)) != len(labels):
        raise ValueError("labels không được chứa nhãn trùng.")

    label_indices = {
        label: index for index, label in enumerate(labels)
    }
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    # labels truyền vào phải bao gồm mọi nhãn xuất hiện trong dữ liệu.
    for actual, predicted in zip(y_true, y_pred):
        if actual not in label_indices or predicted not in label_indices:
            raise ValueError(
                "labels phải chứa mọi nhãn trong y_true và y_pred."
            )

        cm[label_indices[actual], label_indices[predicted]] += 1

    return cm, labels


def accuracy_score(y_true, y_pred):
    """Tính tỷ lệ dự đoán đúng."""
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def _per_class_from_cm(cm, labels):
    """Tính precision, recall, F1 và support từ confusion matrix."""
    tp = np.diag(cm).astype(float)
    predicted_counts = cm.sum(axis=0)
    actual_counts = cm.sum(axis=1)

    # Mẫu số bằng 0 thì trả về 0.
    precision = np.divide(
        tp,
        predicted_counts,
        out=np.zeros_like(tp),
        where=predicted_counts != 0
    )
    recall = np.divide(
        tp,
        actual_counts,
        out=np.zeros_like(tp),
        where=actual_counts != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0
    )

    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(actual_counts[index])
        }
        for index, label in enumerate(labels)
    }


def precision_recall_f1_per_class(y_true, y_pred, labels=None):
    """Trả về metric của từng lớp."""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    return _per_class_from_cm(cm, labels)


def _macro_average(per_class, metric):
    """Lấy trung bình không trọng số giữa các lớp."""
    return float(np.mean([
        values[metric] for values in per_class.values()
    ]))


def precision_score_macro(y_true, y_pred):
    """Tính macro precision."""
    per_class = precision_recall_f1_per_class(y_true, y_pred)
    return _macro_average(per_class, "precision")


def recall_score_macro(y_true, y_pred):
    """Tính macro recall."""
    per_class = precision_recall_f1_per_class(y_true, y_pred)
    return _macro_average(per_class, "recall")


def f1_score_macro(y_true, y_pred):
    """Tính trung bình F1 của từng lớp."""
    per_class = precision_recall_f1_per_class(y_true, y_pred)
    return _macro_average(per_class, "f1")


def evaluate_classification(y_true, y_pred, labels=None):
    """Tổng hợp các metric, chỉ tính confusion matrix một lần."""
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    per_class = _per_class_from_cm(cm, labels)

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": _macro_average(per_class, "precision"),
        "recall_macro": _macro_average(per_class, "recall"),
        "f1_macro": _macro_average(per_class, "f1"),
        "confusion_matrix": cm,
        "labels": labels,
        "per_class": per_class
    }