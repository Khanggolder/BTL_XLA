from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from hog_demo_utils import (
    DEFAULT_HOG_CONFIG,
    apply_condition_transform,
    compute_feature_dim,
    crop_box,
    extract_hog_feature,
    json_ready_config,
    normalize_config,
    read_image,
)


RANDOM_STATE = 126
GROUP_ORDER = ["prohibitory", "danger", "mandatory", "other"]
PROHIBITORY = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
DANGER = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
MANDATORY = [33, 34, 35, 36, 37, 38, 39, 40]
OTHER = [6, 12, 13, 14, 17, 32, 41, 42]
CONDITIONS = ["clean", "low_brightness", "high_brightness", "low_contrast", "gaussian_blur", "gaussian_noise"]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def map_class_group(class_id: int) -> str:
    class_id = int(class_id)
    if class_id in PROHIBITORY:
        return "prohibitory"
    if class_id in DANGER:
        return "danger"
    if class_id in MANDATORY:
        return "mandatory"
    if class_id in OTHER:
        return "other"
    return "unknown"


def load_gtsdb_annotations(gt_path: Path) -> pd.DataFrame:
    columns = ["filename", "xmin", "ymin", "xmax", "ymax", "class_id"]
    df = pd.read_csv(gt_path, sep=";", names=columns)
    df[["xmin", "ymin", "xmax", "ymax", "class_id"]] = df[["xmin", "ymin", "xmax", "ymax", "class_id"]].astype(int)
    df["true_label"] = df["class_id"].apply(map_class_group)
    unknown = sorted(df.loc[df["true_label"] == "unknown", "class_id"].unique())
    if unknown:
        raise ValueError(f"Có ClassID chưa được map nhóm: {unknown}")
    return df.reset_index(drop=True)


def make_filename_split(df: pd.DataFrame, test_size: float = 0.25, random_state: int = RANDOM_STATE) -> tuple[set[str], set[str]]:
    file_df = df[["filename"]].drop_duplicates().reset_index(drop=True)
    filenames = file_df["filename"].to_numpy()
    file_labels = df.groupby("filename")["true_label"].agg(lambda s: s.value_counts().idxmax()).reindex(filenames).to_numpy()
    splitter = GroupShuffleSplit(n_splits=50, test_size=test_size, random_state=random_state)
    all_labels = set(df["true_label"].unique())
    for train_file_idx, test_file_idx in splitter.split(filenames, file_labels, groups=filenames):
        train_files = set(filenames[train_file_idx])
        test_files = set(filenames[test_file_idx])
        train_labels = set(df[df["filename"].isin(train_files)]["true_label"].unique())
        test_labels = set(df[df["filename"].isin(test_files)]["true_label"].unique())
        if train_labels == all_labels and test_labels == all_labels:
            return train_files, test_files
    raise ValueError("Không tạo được filename split có đủ lớp trong train/test.")


def make_linear_svc() -> LinearSVC:
    try:
        return LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=30000, dual="auto")
    except TypeError:
        return LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=30000)


def infer_optimized_config(notebook_path: Path) -> dict[str, Any]:
    config = normalize_config(DEFAULT_HOG_CONFIG)
    if notebook_path.exists():
        text = notebook_path.read_text(encoding="utf-8", errors="ignore")
        strict_match = re.search(
            r"`image_size=\((\d+),\s*(\d+)\)`,\s*"
            r"`orientations=(\d+)`,\s*"
            r"`pixels_per_cell=\((\d+),\s*(\d+)\)`,\s*"
            r"`cells_per_block=\((\d+),\s*(\d+)\)`,\s*"
            r"`preprocessing=([a-z_]+)`",
            text,
        )
        if strict_match:
            config["image_size"] = (int(strict_match.group(1)), int(strict_match.group(2)))
            config["orientations"] = int(strict_match.group(3))
            config["pixels_per_cell"] = (int(strict_match.group(4)), int(strict_match.group(5)))
            config["cells_per_block"] = (int(strict_match.group(6)), int(strict_match.group(7)))
            config["preprocessing"] = "gray_clahe"
            return config
        match = re.search(r"orientations t?t nh?t l?\s+(\d+)", text)
        if match:
            config["orientations"] = int(match.group(1))
            return config
        print("Không infer được optimized config từ notebook, fallback image_size=(96, 96), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2).")
    else:
        print("Không tìm thấy notebook để infer optimized config, fallback image_size=(96, 96), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2).")
    config["image_size"] = (96, 96)
    config["orientations"] = 9
    config["pixels_per_cell"] = (8, 8)
    config["cells_per_block"] = (2, 2)
    config["preprocessing"] = "gray_clahe"
    return config

def load_roi_records(df: pd.DataFrame, data_root: Path) -> tuple[list[np.ndarray], pd.DataFrame]:
    crops: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Load ROI crops"):
        image_path = data_root / row["filename"]
        image = read_image(image_path)
        crop = crop_box(image, (row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
        if crop.size == 0:
            continue
        crops.append(crop)
        rows.append(
            {
                "roi_id": len(rows),
                "source_index": int(idx),
                "filename": row["filename"],
                "image_path": str(Path(row["filename"])),
                "xmin": int(row["xmin"]),
                "ymin": int(row["ymin"]),
                "xmax": int(row["xmax"]),
                "ymax": int(row["ymax"]),
                "class_id": int(row["class_id"]),
                "true_label": row["true_label"],
            }
        )
    return crops, pd.DataFrame(rows)


def extract_features(crops: list[np.ndarray], config: dict[str, Any], desc: str) -> np.ndarray:
    features = [extract_hog_feature(crop, config) for crop in tqdm(crops, desc=desc)]
    return np.vstack(features)


def train_and_evaluate(
    model_name: str,
    role: str,
    config: dict[str, Any],
    crops: list[np.ndarray],
    metadata: pd.DataFrame,
    train_files: set[str],
    test_files: set[str],
) -> tuple[dict[str, Any], Pipeline, np.ndarray, np.ndarray, np.ndarray]:
    X = extract_features(crops, config, f"Extract HOG - {model_name}")
    y = metadata["true_label"].to_numpy()
    filenames = metadata["filename"].to_numpy()
    train_mask = np.isin(filenames, list(train_files))
    test_mask = np.isin(filenames, list(test_files))

    clf = Pipeline([("scaler", StandardScaler()), ("model", make_linear_svc())])
    clf.fit(X[train_mask], y[train_mask])
    y_pred = clf.predict(X[test_mask])
    precision, recall, f1, _ = precision_recall_fscore_support(y[test_mask], y_pred, average="macro", zero_division=0)
    config = normalize_config(config)
    result = {
        "model": model_name,
        "role": role,
        "image_size": str(tuple(config["image_size"])),
        "orientations": int(config["orientations"]),
        "pixels_per_cell": str(tuple(config["pixels_per_cell"])),
        "cells_per_block": str(tuple(config["cells_per_block"])),
        "block_norm": config["block_norm"],
        "preprocessing": config["preprocessing"],
        "accuracy": accuracy_score(y[test_mask], y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "feature_dim": int(X.shape[1]),
    }
    return result, clf, y[test_mask], y_pred, test_mask


def evaluate_under_condition(
    model_name: str,
    clf: Pipeline,
    config: dict[str, Any],
    test_crops: list[np.ndarray],
    y_true: np.ndarray,
    condition: str,
) -> dict[str, Any]:
    transformed = [
        apply_condition_transform(crop, condition, seed=RANDOM_STATE + idx)
        for idx, crop in enumerate(test_crops)
    ]
    X = extract_features(transformed, config, f"Robustness {model_name} - {condition}")
    y_pred = clf.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "model": model_name,
        "condition": condition,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def make_robustness_summary(robustness_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in robustness_results.groupby("model", sort=False):
        clean_macro_f1 = group.loc[group["condition"] == "clean", "macro_f1"].iloc[0]
        non_clean = group[group["condition"] != "clean"]
        avg_robust_macro_f1 = non_clean["macro_f1"].mean()
        rows.append(
            {
                "model": model_name,
                "clean_macro_f1": clean_macro_f1,
                "avg_robust_macro_f1": avg_robust_macro_f1,
                "worst_case_macro_f1": non_clean["macro_f1"].min(),
                "drop_from_clean": clean_macro_f1 - avg_robust_macro_f1,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HOG + SVM demo artifacts for Streamlit.")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data/FullIJCNN2013"))
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--notebook", default="traffic_sign_cv.ipynb")
    args = parser.parse_args()

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    data_root = Path(args.data_root)
    artifact_dir = Path(args.artifact_dir)
    gt_path = data_root / "gt.txt"
    if not data_root.exists() or not gt_path.exists():
        raise FileNotFoundError("Cần đặt GTSDB tại data/FullIJCNN2013/ hoặc truyền --data-root/biến môi trường DATA_ROOT.")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = load_gtsdb_annotations(gt_path)
    train_files, test_files = make_filename_split(df)
    crops, metadata = load_roi_records(df, data_root)

    baseline_config = normalize_config(DEFAULT_HOG_CONFIG)
    optimized_config = infer_optimized_config(Path(args.notebook))

    baseline_result, baseline_clf, baseline_y_true, baseline_y_pred, test_mask = train_and_evaluate(
        "Baseline HOG + SVM",
        "baseline chính",
        baseline_config,
        crops,
        metadata,
        train_files,
        test_files,
    )
    optimized_result, optimized_clf, optimized_y_true, optimized_y_pred, _ = train_and_evaluate(
        "Optimized HOG + SVM",
        "cải tiến chính",
        optimized_config,
        crops,
        metadata,
        train_files,
        test_files,
    )

    test_metadata = metadata.loc[test_mask].reset_index(drop=True).copy()
    test_metadata["baseline_pred"] = baseline_y_pred
    test_metadata["optimized_pred"] = optimized_y_pred
    test_metadata["split"] = "test"

    final_metrics = pd.DataFrame([baseline_result, optimized_result])
    baseline_f1 = float(final_metrics.loc[final_metrics["model"] == "Baseline HOG + SVM", "macro_f1"].iloc[0])
    final_metrics["delta_vs_baseline"] = final_metrics["macro_f1"] - baseline_f1

    test_crops = [crop for crop, is_test in zip(crops, test_mask) if is_test]
    y_true_test = test_metadata["true_label"].to_numpy()
    robustness_rows = []
    for condition in CONDITIONS:
        robustness_rows.append(evaluate_under_condition("Baseline HOG + SVM", baseline_clf, baseline_config, test_crops, y_true_test, condition))
        robustness_rows.append(evaluate_under_condition("Optimized HOG + SVM", optimized_clf, optimized_config, test_crops, y_true_test, condition))
    robustness_results = pd.DataFrame(robustness_rows)
    robustness_summary = make_robustness_summary(robustness_results)

    _, _, baseline_per_class_f1, _ = precision_recall_fscore_support(
        baseline_y_true,
        baseline_y_pred,
        labels=GROUP_ORDER,
        average=None,
        zero_division=0,
    )
    _, _, optimized_per_class_f1, _ = precision_recall_fscore_support(
        optimized_y_true,
        optimized_y_pred,
        labels=GROUP_ORDER,
        average=None,
        zero_division=0,
    )
    per_class_f1 = pd.DataFrame(
        {
            "class": GROUP_ORDER,
            "baseline_f1": baseline_per_class_f1,
            "optimized_f1": optimized_per_class_f1,
        }
    )
    per_class_f1["delta_f1"] = per_class_f1["optimized_f1"] - per_class_f1["baseline_f1"]

    error_examples = test_metadata.copy()
    error_examples["case_type"] = "same_status"
    fixed_mask = (error_examples["baseline_pred"] != error_examples["true_label"]) & (
        error_examples["optimized_pred"] == error_examples["true_label"]
    )
    regressed_mask = (error_examples["baseline_pred"] == error_examples["true_label"]) & (
        error_examples["optimized_pred"] != error_examples["true_label"]
    )
    error_examples.loc[fixed_mask, "case_type"] = "baseline_wrong_optimized_correct"
    error_examples.loc[regressed_mask, "case_type"] = "baseline_correct_optimized_wrong"
    error_examples = error_examples[error_examples["case_type"] != "same_status"].reset_index(drop=True)
    if not error_examples.empty:
        error_examples["priority"] = (error_examples["true_label"] == "mandatory").astype(int)
        error_examples = error_examples.sort_values(["case_type", "priority"], ascending=[True, False]).drop(columns=["priority"])

    joblib.dump(baseline_clf, artifact_dir / "baseline_model.joblib")
    joblib.dump(optimized_clf, artifact_dir / "optimized_model.joblib")
    test_metadata.to_csv(artifact_dir / "test_metadata.csv", index=False)
    final_metrics.to_csv(artifact_dir / "final_metrics.csv", index=False)
    robustness_results.to_csv(artifact_dir / "robustness_results.csv", index=False)
    robustness_summary.to_csv(artifact_dir / "robustness_summary.csv", index=False)
    error_examples.to_csv(artifact_dir / "error_examples.csv", index=False)
    per_class_f1.to_csv(artifact_dir / "per_class_f1.csv", index=False)

    configs = {
        "random_state": RANDOM_STATE,
        "labels": GROUP_ORDER,
        "data_root": str(data_root),
        "baseline": json_ready_config(baseline_config),
        "optimized": json_ready_config(optimized_config),
        "feature_dim": {
            "baseline": compute_feature_dim(baseline_config),
            "optimized": compute_feature_dim(optimized_config),
        },
    }
    (artifact_dir / "configs.json").write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nĐã export artifacts:")
    for path in sorted(artifact_dir.iterdir()):
        print(f"- {path}")
    print("\nKết quả chính:")
    print(final_metrics[["model", "macro_f1", "feature_dim", "orientations", "pixels_per_cell", "cells_per_block", "preprocessing"]].to_string(index=False))


if __name__ == "__main__":
    main()

