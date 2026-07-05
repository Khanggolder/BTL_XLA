from __future__ import annotations
#streamlit run app.py 
import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from hog_demo_utils import (
    DEFAULT_HOG_CONFIG,
    apply_condition_transform,
    compute_feature_dim,
    compute_hog_visualization,
    cell_histogram,
    compute_gradients,
    crop_box,
    draw_bbox,
    draw_cell_on_preprocessed,
    extract_hog_feature,
    feature_dim_breakdown,
    make_gradient_orientation_display,
    make_orientation_bin_figure,
    normalize_config,
    predict_roi,
    preprocess_for_hog,
    read_image,
)


ARTIFACT_DIR = Path("artifacts")
REQUIRED_ARTIFACTS = [
    "baseline_model.joblib",
    "optimized_model.joblib",
    "configs.json",
    "test_metadata.csv",
    "final_metrics.csv",
    "robustness_results.csv",
    "robustness_summary.csv",
    "error_examples.csv",
    "per_class_f1.csv",
]
GROUP_ORDER = ["prohibitory", "danger", "mandatory", "other"]
CONDITIONS = ["clean", "low_brightness", "high_brightness", "low_contrast", "gaussian_blur", "gaussian_noise"]
RANDOM_STATE = 126


st.set_page_config(page_title="HOG + SVM GTSDB Demo", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    .metric-card {
        border: 1px solid #e6e8eb;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: white;
        color: #111827;
    }
    .metric-card b {
        color: #0f172a;
    }
    .pipeline-card {
        margin: 1rem 0 1.1rem 0;
        border: 1px solid #e6e8eb;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: white;
        color: #111827;
    }
    .pipeline-card b {
        color: #0f172a;
    }
    .pipeline-flow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.45rem;
        margin-top: 0.65rem;
    }
    .pipeline-flow span {
        border: 1px solid #d1d5db;
        border-radius: 6px;
        padding: 0.45rem 0.65rem;
        background: #f8fafc;
        font-weight: 600;
        white-space: nowrap;
    }
    .pipeline-flow em {
        color: #64748b;
        font-style: normal;
        font-weight: 700;
    }
    .good-box {
        border-left: 5px solid #23a55a;
        padding: 0.7rem 0.9rem;
        background: #eefaf2;
        border-radius: 6px;
        color: #111827;
    }
    .warn-box {
        border-left: 5px solid #f59f00;
        padding: 0.7rem 0.9rem;
        background: #fff7e6;
        border-radius: 6px;
        color: #111827;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #eef0f2; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def missing_artifacts() -> list[str]:
    return [name for name in REQUIRED_ARTIFACTS if not (ARTIFACT_DIR / name).exists()]


@st.cache_resource
def load_models(artifact_dir: str):
    path = Path(artifact_dir)
    return joblib.load(path / "baseline_model.joblib"), joblib.load(path / "optimized_model.joblib")


@st.cache_data
def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    file = Path(path)
    if file.exists():
        return pd.read_csv(file)
    return pd.DataFrame()


def format_float_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["float", "float64"]).columns:
        out[col] = out[col].map(lambda v: f"{v:.4f}")
    return out


def get_data_root(default_from_config: str | None = None) -> Path:
    default = os.environ.get("DATA_ROOT") or default_from_config or "data/FullIJCNN2013"
    return Path(
        st.sidebar.text_input(
            "DATA_ROOT",
            value=default,
            help="Có thể đổi nếu data/FullIJCNN2013 nằm ở vị trí khác.",
        )
    )


def show_missing_guide(missing: list[str]) -> None:
    st.title("Demo HOG + SVM cho GTSDB")
    st.warning("Chưa tìm thấy artifacts cần thiết.")
    st.write("Hãy chạy export trước, Streamlit app không train lại model khi mở demo.")
    st.code("python export_artifacts.py --data-root data/FullIJCNN2013 --notebook traffic_sign_cv.ipynb\nstreamlit run streamlit_app.py", language="bash")
    st.write("Thiếu file:")
    st.write(", ".join(missing))


def load_roi_from_row(row: pd.Series, data_root: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        image = read_image(data_root / str(row["filename"]))
        box = (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))
        return image, crop_box(image, box)
    except Exception:
        return None, None


def row_label(row: pd.Series) -> str:
    return (
        f"#{int(row['roi_id'])} | {row['filename']} | true={row['true_label']} | "
        f"B={row['baseline_pred']} | O={row['optimized_pred']}"
    )


def selected_row(test_metadata: pd.DataFrame) -> pd.Series | None:
    if test_metadata.empty:
        return None
    roi_id = st.session_state.get("selected_roi_id")
    if roi_id is None or roi_id not in set(test_metadata["roi_id"].astype(int)):
        return test_metadata.iloc[0]
    return test_metadata[test_metadata["roi_id"].astype(int) == int(roi_id)].iloc[0]




def _draw_rect_rgb(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], thickness: int = 1) -> None:
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(x1)))
    x2 = max(0, min(width - 1, int(x2)))
    y1 = max(0, min(height - 1, int(y1)))
    y2 = max(0, min(height - 1, int(y2)))
    for offset in range(max(1, int(thickness))):
        if y1 + offset <= y2:
            image[y1 + offset, x1 : x2 + 1] = color
        if y2 - offset >= y1:
            image[y2 - offset, x1 : x2 + 1] = color
        if x1 + offset <= x2:
            image[y1 : y2 + 1, x1 + offset] = color
        if x2 - offset >= x1:
            image[y1 : y2 + 1, x2 - offset] = color


def draw_hog_grid_overlay(
    preprocessed_gray: np.ndarray,
    cell_x: int,
    cell_y: int,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    cell_w, cell_h = pixels_per_cell
    block_w, block_h = cells_per_block
    height, width = preprocessed_gray.shape[:2]
    out = np.repeat(preprocessed_gray.astype(np.uint8)[..., None], 3, axis=2)

    grid_color = (148, 163, 184)
    for x in range(cell_w, width, cell_w):
        out[:, x : min(x + 1, width)] = grid_color
    for y in range(cell_h, height, cell_h):
        out[y : min(y + 1, height), :] = grid_color

    n_cells_x = max(1, width // cell_w)
    n_cells_y = max(1, height // cell_h)
    block_x = min(max(0, int(cell_x)), max(0, n_cells_x - block_w))
    block_y = min(max(0, int(cell_y)), max(0, n_cells_y - block_h))

    _draw_rect_rgb(
        out,
        block_x * cell_w,
        block_y * cell_h,
        min((block_x + block_w) * cell_w, width) - 1,
        min((block_y + block_h) * cell_h, height) - 1,
        (34, 197, 94),
        thickness=2,
    )
    _draw_rect_rgb(
        out,
        int(cell_x) * cell_w,
        int(cell_y) * cell_h,
        min((int(cell_x) + 1) * cell_w, width) - 1,
        min((int(cell_y) + 1) * cell_h, height) - 1,
        (255, 80, 40),
        thickness=2,
    )
    return out, (block_x, block_y)


def render_prediction_badge(label: str, pred: str) -> str:
    return "Đúng" if pred == label else "Sai"


def make_config(image_size: str, orientations: int, pixels_per_cell: str, cells_per_block: str, preprocessing: str):
    size = tuple(int(x) for x in image_size.split("x"))
    ppc = tuple(int(x) for x in pixels_per_cell.split("x"))
    cpb = tuple(int(x) for x in cells_per_block.split("x"))
    return normalize_config(
        {
            "image_size": size,
            "orientations": orientations,
            "pixels_per_cell": ppc,
            "cells_per_block": cpb,
            "block_norm": "L2-Hys",
            "preprocessing": preprocessing,
        }
    )




def pair_label(value: tuple[int, int] | list[int]) -> str:
    return f"{int(value[0])}x{int(value[1])}"


def configs_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left = normalize_config(left)
    right = normalize_config(right)
    keys = ["image_size", "orientations", "pixels_per_cell", "cells_per_block", "preprocessing"]
    return all(left[key] == right[key] for key in keys)




def config_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    config = normalize_config(config)
    return (
        config["image_size"],
        int(config["orientations"]),
        config["pixels_per_cell"],
        config["cells_per_block"],
        config["preprocessing"],
    )
def tab4_config_from_state() -> dict[str, Any]:
    return make_config(
        st.session_state["tab4_image_size"],
        int(st.session_state["tab4_orientations"]),
        st.session_state["tab4_pixels_per_cell"],
        st.session_state["tab4_cells_per_block"],
        st.session_state["tab4_preprocessing"],
    )

PROHIBITORY = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
DANGER = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
MANDATORY = [33, 34, 35, 36, 37, 38, 39, 40]
OTHER = [6, 12, 13, 14, 17, 32, 41, 42]


def map_class_group(class_id: int) -> str:
    if int(class_id) in PROHIBITORY:
        return "prohibitory"
    if int(class_id) in DANGER:
        return "danger"
    if int(class_id) in MANDATORY:
        return "mandatory"
    if int(class_id) in OTHER:
        return "other"
    return "unknown"


@st.cache_data(show_spinner=False)
def custom_config_evaluation(data_root_str: str, config: dict[str, Any], sample_size: str) -> pd.DataFrame:
    data_root = Path(data_root_str)
    gt_path = data_root / "gt.txt"
    columns = ["filename", "xmin", "ymin", "xmax", "ymax", "class_id"]
    df = pd.read_csv(gt_path, sep=";", names=columns)
    df["true_label"] = df["class_id"].apply(map_class_group)
    filenames = df[["filename"]].drop_duplicates()["filename"].to_numpy()
    file_labels = df.groupby("filename")["true_label"].agg(lambda s: s.value_counts().idxmax()).reindex(filenames).to_numpy()
    splitter = GroupShuffleSplit(n_splits=50, test_size=0.25, random_state=RANDOM_STATE)
    train_files, test_files = None, None
    all_labels = set(df["true_label"].unique())
    for train_idx, test_idx in splitter.split(filenames, file_labels, groups=filenames):
        tr = set(filenames[train_idx])
        te = set(filenames[test_idx])
        if set(df[df["filename"].isin(tr)]["true_label"].unique()) == all_labels and set(df[df["filename"].isin(te)]["true_label"].unique()) == all_labels:
            train_files, test_files = tr, te
            break
    if train_files is None or test_files is None:
        raise ValueError("Không tạo được split.")

    train_df = df[df["filename"].isin(train_files)].reset_index(drop=True)
    test_df = df[df["filename"].isin(test_files)].reset_index(drop=True)
    if sample_size != "full":
        n = min(int(sample_size), len(test_df))
        test_df = test_df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    def build(rows: pd.DataFrame, desc: str):
        X, y = [], []
        for _, row in rows.iterrows():
            image = read_image(data_root / row["filename"])
            roi = crop_box(image, (row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
            X.append(extract_hog_feature(roi, config))
            y.append(row["true_label"])
        return np.vstack(X), np.array(y)

    X_train, y_train = build(train_df, "train")
    X_test, y_test = build(test_df, "test")
    try:
        svm = LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=30000, dual="auto")
    except TypeError:
        svm = LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=30000)
    clf = Pipeline([("scaler", StandardScaler()), ("model", svm)])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    return pd.DataFrame(
        [
            {
                "sample_size": len(test_df),
                "accuracy": accuracy_score(y_test, pred),
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "feature_dim": X_train.shape[1],
            }
        ]
    )


missing = missing_artifacts()
if missing:
    show_missing_guide(missing)
    st.stop()

configs = load_json(str(ARTIFACT_DIR / "configs.json"))
baseline_config = normalize_config(configs["baseline"])
optimized_config = normalize_config(configs["optimized"])
baseline_model, optimized_model = load_models(str(ARTIFACT_DIR))
test_metadata = load_csv(str(ARTIFACT_DIR / "test_metadata.csv"))
final_metrics = load_csv(str(ARTIFACT_DIR / "final_metrics.csv"))
robustness_results = load_csv(str(ARTIFACT_DIR / "robustness_results.csv"))
robustness_summary = load_csv(str(ARTIFACT_DIR / "robustness_summary.csv"))
error_examples = load_csv(str(ARTIFACT_DIR / "error_examples.csv"))
per_class_f1 = load_csv(str(ARTIFACT_DIR / "per_class_f1.csv"))

tab4_defaults = {
    "tab4_image_size": pair_label(optimized_config["image_size"]),
    "tab4_orientations": int(optimized_config["orientations"]),
    "tab4_pixels_per_cell": pair_label(optimized_config["pixels_per_cell"]),
    "tab4_cells_per_block": pair_label(optimized_config["cells_per_block"]),
    "tab4_preprocessing": optimized_config["preprocessing"],
}
for key, value in tab4_defaults.items():
    st.session_state.setdefault(key, value)


st.sidebar.title("HOG + SVM Demo")
data_root = get_data_root(configs.get("data_root"))
quick_demo = st.sidebar.toggle("Quick demo", value=True)
st.sidebar.caption("Quick demo giữ mọi phần nặng sau nút bấm và chỉ xử lý ROI đang chọn.")

st.title("Phân loại nhóm biển báo GTSDB bằng HOG + SVM")
st.caption("Demo tập trung vào giả thuyết: tối ưu cấu hình HOG giúp cải thiện Macro F1 so với baseline.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "1. Tổng quan",
        "2. Phân loại ROI",
        "3. Trực quan HOG",
        "4. Thử tham số",
        "5. Robustness",
        "6. Lỗi sai",
        "7. Slide mode",
    ]
)

with tab1:
    st.subheader("Tổng quan giả thuyết")
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown('<div class="metric-card"><b>Input</b><br>ROI biển báo crop từ ground-truth bbox.</div>', unsafe_allow_html=True)
    col_b.markdown('<div class="metric-card"><b>Output</b><br>prohibitory, danger, mandatory, other.</div>', unsafe_allow_html=True)
    col_c.markdown('<div class="metric-card"><b>Giả thuyết</b><br>Tối ưu HOG cải thiện Macro F1.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="pipeline-card">
            <b>Pipeline xử lý</b>
            <div class="pipeline-flow">
                <span>GTSDB + ground-truth bbox</span>
                <em>-&gt;</em>
                <span>Crop ROI biển báo</span>
                <em>-&gt;</em>
                <span>Preprocessing ảnh</span>
                <em>-&gt;</em>
                <span>Trích xuất HOG</span>
                <em>-&gt;</em>
                <span>StandardScaler + Linear SVM</span>
                <em>-&gt;</em>
                <span>Dự đoán nhóm + Macro F1</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    baseline_row = final_metrics[final_metrics["model"] == "Baseline HOG + SVM"].iloc[0]
    optimized_row = final_metrics[final_metrics["model"] == "Optimized HOG + SVM"].iloc[0]
    tab4_config = tab4_config_from_state()
    tab4_is_exported_optimized = configs_equal(tab4_config, optimized_config)
    tab4_eval_signature = st.session_state.get("tab4_eval_signature")
    tab4_eval_macro_f1 = st.session_state.get("tab4_eval_macro_f1")
    if tab4_is_exported_optimized:
        tab4_macro_f1 = f"{float(optimized_row['macro_f1']):.4f}"
    elif tab4_eval_signature == config_signature(tab4_config) and tab4_eval_macro_f1 is not None:
        tab4_macro_f1 = f"{float(tab4_eval_macro_f1):.4f}"
    else:
        tab4_macro_f1 = "chưa đánh giá"
    overview_metrics = pd.DataFrame(
        [
            {
                "model": "Baseline HOG + SVM",
                "image_size": pair_label(baseline_config["image_size"]),
                "orientations": int(baseline_config["orientations"]),
                "pixels_per_cell": pair_label(baseline_config["pixels_per_cell"]),
                "cells_per_block": pair_label(baseline_config["cells_per_block"]),
                "preprocessing": baseline_config["preprocessing"],
                "feature_dim": compute_feature_dim(baseline_config),
                "macro_f1": f"{float(baseline_row['macro_f1']):.4f}",
            },
            {
                "model": "Optimized HOG + SVM (theo Tab 4)",
                "image_size": pair_label(tab4_config["image_size"]),
                "orientations": int(tab4_config["orientations"]),
                "pixels_per_cell": pair_label(tab4_config["pixels_per_cell"]),
                "cells_per_block": pair_label(tab4_config["cells_per_block"]),
                "preprocessing": tab4_config["preprocessing"],
                "feature_dim": compute_feature_dim(tab4_config),
                "macro_f1": tab4_macro_f1,
            },
        ]
    )
    st.dataframe(overview_metrics, width='stretch', hide_index=True)
    st.markdown(
        "Dòng **Baseline** giữ nguyên. Dòng **Optimized** lấy tham số đang chọn ở **Tab 4 - Thử tham số**; "
        "nếu đổi tham số thì `feature_dim` cập nhật ngay, còn `macro_f1` chỉ có sau khi chạy đánh giá."
    )

with tab2:
    st.subheader("Demo phân loại ROI")
    chosen = None
    left, right = st.columns([1, 1])
    with left:
        label_filter = st.selectbox("Filter true label", ["all"] + GROUP_ORDER)
        prefer_fixed = st.checkbox('Chỉ hiện "baseline sai nhưng optimized đúng"', value=True)
        view_df = test_metadata.copy()
        if label_filter != "all":
            view_df = view_df[view_df["true_label"] == label_filter]
        if prefer_fixed:
            view_df = view_df[(view_df["baseline_pred"] != view_df["true_label"]) & (view_df["optimized_pred"] == view_df["true_label"])]
        if view_df.empty:
            st.info("Không có mẫu phù hợp với filter hiện tại.")
        else:
            labels = [row_label(row) for _, row in view_df.iterrows()]
            current_default = 0
            if "selected_roi_id" in st.session_state:
                ids = view_df["roi_id"].astype(int).tolist()
                if int(st.session_state["selected_roi_id"]) in ids:
                    current_default = ids.index(int(st.session_state["selected_roi_id"]))
            choice = st.selectbox("Chọn ảnh test", labels, index=current_default)
            chosen = view_df.iloc[labels.index(choice)]
            st.session_state["selected_roi_id"] = int(chosen["roi_id"])

    with right:
        row = chosen
        if row is None:
            st.info("Không có ảnh phù hợp để hiển thị.")
        else:
            full_image, roi = load_roi_from_row(row, data_root)
            display_roi = roi
            if full_image is not None:
                boxed = draw_bbox(full_image, (row["xmin"], row["ymin"], row["xmax"], row["ymax"]), row["true_label"])
                st.image(boxed, caption="Ảnh gốc với ground-truth bbox", width='stretch')
            else:
                st.warning("Không đọc được ảnh gốc từ DATA_ROOT.")

            if display_roi is not None:
                st.image(display_roi, caption="ROI crop", width=260)
                true_label = str(row["true_label"])
                base_pred = predict_roi(baseline_model, display_roi, baseline_config)
                opt_pred = predict_roi(optimized_model, display_roi, optimized_config)
                c1, c2, c3 = st.columns(3)
                c1.metric("True label", true_label)
                c2.metric("Baseline", base_pred)
                c3.metric("Optimized", opt_pred)
                if base_pred != true_label and opt_pred == true_label:
                    st.markdown('<div class="good-box">Đây là ví dụ optimized sửa được lỗi của baseline.</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("Gi\u1ea3i th\u00edch tr\u1ef1c quan HOG th\u01b0\u1eddng vs HOG t\u1ed1i \u01b0u")
    row = selected_row(test_metadata)
    _, roi = load_roi_from_row(row, data_root) if row is not None else (None, None)
    if roi is None:
        st.warning("Không có ROI để hiển thị. Hãy chọn ảnh test ở Tab 2 hoặc kiểm tra DATA_ROOT.")
    else:
        baseline_orientation = int(baseline_config["orientations"])
        tab4_config = tab4_config_from_state()
        tab4_is_exported_optimized = configs_equal(tab4_config, optimized_config)
        optimized_label = "Optimized đã export" if tab4_is_exported_optimized else "Cấu hình Tab 4"
        optimized_orientation = int(tab4_config["orientations"])
        baseline_dim = compute_feature_dim(baseline_config)
        optimized_dim = compute_feature_dim(tab4_config)
        baseline_breakdown = feature_dim_breakdown(baseline_config)
        optimized_breakdown = feature_dim_breakdown(tab4_config)
        degrees_per_bin_baseline = 180 / baseline_orientation
        degrees_per_bin_optimized = 180 / optimized_orientation
        cell_w, cell_h = baseline_config["pixels_per_cell"]

        base_pre, base_hog = compute_hog_visualization(roi, baseline_config)
        _, opt_hog = compute_hog_visualization(roi, tab4_config)
        _, _, magnitude, orientation_deg = compute_gradients(base_pre)
        magnitude_display = magnitude / (float(magnitude.max()) + 1e-8)
        orientation_display = make_gradient_orientation_display(orientation_deg, magnitude)

        visual_cols = st.columns(3)
        visual_cols[0].image(roi, caption="ROI g\u1ed1c", width='stretch')
        visual_cols[1].image(base_pre, caption=f"ROI sau preprocessing: {baseline_config['preprocessing']}", width='stretch', clamp=True)
        visual_cols[2].image(magnitude_display, caption="Gradient magnitude", width='stretch', clamp=True)

        visual_cols_2 = st.columns(3)
        visual_cols_2[0].image(orientation_display, caption="Gradient orientation 0-180\u00b0", width='stretch')
        visual_cols_2[1].image(base_hog, caption=f"HOG baseline, orientations = {baseline_orientation}", width='stretch', clamp=True)
        visual_cols_2[2].image(opt_hog, caption=f"HOG {optimized_label}, orientations = {optimized_orientation}", width='stretch', clamp=True)

        st.info(
            f"Phần HOG bên phải đang dùng {optimized_label}. "
            f"Cấu hình: image_size={tab4_config['image_size']}, orientations={optimized_orientation}, "
            f"pixels_per_cell={tab4_config['pixels_per_cell']}, cells_per_block={tab4_config['cells_per_block']}, "
            f"preprocessing={tab4_config['preprocessing']}, feature_dim={optimized_dim}."
        )
        if optimized_orientation == baseline_orientation:
            orientation_note = f"Baseline và {optimized_label} dùng cùng số orientation bins; khác biệt chính nằm ở các tham số HOG khác như image_size/preprocessing."
        elif optimized_orientation < baseline_orientation:
            orientation_note = f"{optimized_label} dùng ít orientation bins hơn nên hướng cạnh được gom thô hơn."
        else:
            orientation_note = f"{optimized_label} dùng nhiều orientation bins hơn nên hướng cạnh được chia mịn hơn."
        st.markdown(
            f"""
            - HOG không nhìn màu chính, mà nhìn gradient/cạnh.
            - Mỗi cell gồm các gradient thành histogram hướng.
            - Baseline {baseline_orientation} bins: mỗi bin khoảng {degrees_per_bin_baseline:.1f} độ.
            - {optimized_label} {optimized_orientation} bins: mỗi bin khoảng {degrees_per_bin_optimized:.1f} độ.
            - {orientation_note}
            - Nếu đổi tham số ở Tab 4, HOG visualization và feature_dim ở đây đổi theo cấu hình đó.
            """
        )

        wheel_col, formula_col = st.columns([1, 1.1])
        with wheel_col:
            st.pyplot(
                make_orientation_bin_figure(
                    baseline_bins=baseline_orientation,
                    optimized_bins=optimized_orientation,
                ),
                width='stretch',
            )
        with formula_col:
            st.code(
                """n_cells_x = image_width / pixels_per_cell_x
n_cells_y = image_height / pixels_per_cell_y
n_blocks_x = n_cells_x - cells_per_block_x + 1
n_blocks_y = n_cells_y - cells_per_block_y + 1
feature_dim = n_blocks_x * n_blocks_y * cells_per_block_x * cells_per_block_y * orientations""",
                language="text",
            )

        def render_feature_dim_breakdown(name: str, bd: dict[str, Any]) -> None:
            image_w, image_h = bd["image_size"]
            cell_x, cell_y = bd["pixels_per_cell"]
            block_x, block_y = bd["cells_per_block"]
            st.markdown(f"**{name}:**")
            st.write(
                f"{image_w}x{image_h}, pixels_per_cell = {cell_x}x{cell_y} -> "
                f"{bd['n_cells_x']}x{bd['n_cells_y']} cells"
            )
            st.write(
                f"cells_per_block = {block_x}x{block_y} -> "
                f"{bd['n_blocks_x']}x{bd['n_blocks_y']} blocks"
            )
            st.write(
                f"feature_dim = {bd['n_blocks_x']} * {bd['n_blocks_y']} * "
                f"{block_x} * {block_y} * {bd['orientations']} = {bd['feature_dim']}"
            )

        left_dim, right_dim = st.columns(2)
        with left_dim:
            render_feature_dim_breakdown("Baseline", baseline_breakdown)
        with right_dim:
            render_feature_dim_breakdown(optimized_label, optimized_breakdown)
        if optimized_dim < baseline_dim:
            dim_note = f"{optimized_label} giảm feature_dim từ {baseline_dim} xuống {optimized_dim}."
        elif optimized_dim > baseline_dim:
            dim_note = f"{optimized_label} tăng feature_dim từ {baseline_dim} lên {optimized_dim} để giữ nhiều chi tiết hơn."
        else:
            dim_note = f"Baseline và {optimized_label} có cùng feature_dim = {baseline_dim}."
        st.markdown(
            f'<div class="good-box">{dim_note} Đây là trade-off giữa độ mạnh của biểu diễn HOG và độ dài vector đặc trưng.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Histogram hướng gradient trong một cell**")
        hist_control_1, hist_control_2, hist_control_3 = st.columns(3)
        ppc_label = hist_control_1.selectbox("pixels_per_cell", ["2x2", "4x4", "8x8", "12x12", "16x16"], index=2, key="tab3_pixels_per_cell")
        selected_pixels_per_cell = tuple(int(x) for x in ppc_label.split("x"))
        cell_w, cell_h = selected_pixels_per_cell
        n_cells_x = max(1, base_pre.shape[1] // cell_w)
        n_cells_y = max(1, base_pre.shape[0] // cell_h)
        max_cell_x = max(0, n_cells_x - 1)
        max_cell_y = max(0, n_cells_y - 1)
        cell_x = hist_control_2.slider("cell_x", 0, max_cell_x, min(3, max_cell_x))
        cell_y = hist_control_3.slider("cell_y", 0, max_cell_y, min(3, max_cell_y))
        selected_cell, selected_block = draw_hog_grid_overlay(
            base_pre,
            cell_x,
            cell_y,
            selected_pixels_per_cell,
            baseline_config["cells_per_block"],
        )
        base_centers, base_hist = cell_histogram(base_pre, cell_x, cell_y, selected_pixels_per_cell, baseline_orientation)
        opt_centers, opt_hist = cell_histogram(base_pre, cell_x, cell_y, selected_pixels_per_cell, optimized_orientation)

        hist_fig, hist_axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
        hist_axes[0].bar(base_centers, base_hist, width=180 / baseline_orientation * 0.85, color="#4c78a8")
        hist_axes[0].set_title(f"Baseline: {baseline_orientation} bins")
        hist_axes[0].set_xlabel("Hướng gradient (độ)")
        hist_axes[0].set_ylabel("Tổng magnitude")
        hist_axes[0].set_xlim(0, 180)
        hist_axes[1].bar(opt_centers, opt_hist, width=180 / optimized_orientation * 0.85, color="#59a14f")
        hist_axes[1].set_title(f"{optimized_label}: {optimized_orientation} bins")
        hist_axes[1].set_xlabel("Hướng gradient (độ)")
        hist_axes[1].set_xlim(0, 180)
        hist_fig.tight_layout()

        cell_col, hist_col = st.columns([0.85, 1.35])
        with cell_col:
            st.image(
                selected_cell,
                caption=(
                    f"Lưới cell {cell_w}x{cell_h}; cam=cell ({cell_x}, {cell_y}), "
                    f"xanh=block {baseline_config['cells_per_block']} tại {selected_block}"
                ),
                width='stretch',
            )
        with hist_col:
            st.pyplot(hist_fig, width='stretch')
        if optimized_orientation == baseline_orientation:
            hist_note = (
                f"Baseline và {optimized_label} đều chia hướng gradient thành {baseline_orientation} bins. "
                "Histogram orientation giống về số bin; khi đổi pixels_per_cell thì vùng lấy histogram lớn/nhỏ khác nhau."
            )
        else:
            hist_note = (
                f"Baseline dùng {baseline_orientation} bins, {optimized_label} dùng {optimized_orientation} bins. "
                "Số bins khác nhau làm histogram hướng gradient mịn hơn hoặc thô hơn."
            )
        st.markdown(
            hist_note + " Với biển báo giao thông, các đặc trưng quan trọng thường là viền tròn, tam giác, mũi tên và ký hiệu lớn."
        )

with tab4:
    st.subheader("Thử tham số HOG tương tác")
    row = selected_row(test_metadata)
    _, roi = load_roi_from_row(row, data_root) if row is not None else (None, None)

    image_size_options = ["32x32", "48x48", "64x64", "80x80", "96x96", "128x128"]
    pixels_per_cell_options = ["2x2", "4x4", "8x8", "12x12", "16x16"]
    cells_per_block_options = ["1x1", "2x2", "3x3", "4x4"]
    preprocessing_options = ["gray", "gray_clahe", "gray_clahe_blur", "yuv_y", "hsv_v"]

    with st.form("hog_params"):
        c1, c2, c3, c4, c5 = st.columns(5)
        image_size = c1.selectbox("image_size", image_size_options, index=image_size_options.index(st.session_state["tab4_image_size"]))
        orientations = c2.slider("orientations", min_value=1, max_value=18, value=int(st.session_state["tab4_orientations"]))
        pixels_per_cell = c3.selectbox("pixels_per_cell", pixels_per_cell_options, index=pixels_per_cell_options.index(st.session_state["tab4_pixels_per_cell"]))
        cells_per_block = c4.selectbox("cells_per_block", cells_per_block_options, index=cells_per_block_options.index(st.session_state["tab4_cells_per_block"]))
        preprocessing = c5.selectbox("preprocessing", preprocessing_options, index=preprocessing_options.index(st.session_state["tab4_preprocessing"]))
        submitted = st.form_submit_button("Cập nhật visualization và bảng Tab 1")

    if submitted:
        st.session_state["tab4_image_size"] = image_size
        st.session_state["tab4_orientations"] = int(orientations)
        st.session_state["tab4_pixels_per_cell"] = pixels_per_cell
        st.session_state["tab4_cells_per_block"] = cells_per_block
        st.session_state["tab4_preprocessing"] = preprocessing
        st.rerun()

    custom_config = make_config(image_size, orientations, pixels_per_cell, cells_per_block, preprocessing)
    dim = compute_feature_dim(custom_config)
    st.metric("feature_dim", dim if dim > 0 else "Không hợp lệ")
    if roi is not None and dim > 0:
        custom_pre, custom_hog = compute_hog_visualization(roi, custom_config)
        col1, col2, col3 = st.columns(3)
        col1.image(roi, caption="ROI", width='stretch')
        col2.image(custom_pre, caption=f"Preprocessing: {preprocessing}", width='stretch', clamp=True)
        col3.image(custom_hog, caption="HOG visualization", width='stretch', clamp=True)
    elif dim == 0:
        st.warning("Cấu hình này không tạo được block HOG vì cell/block quá lớn so với image_size.")
    else:
        st.warning("Không có ROI để hiển thị.")

    st.markdown(
        """
        - cell nhỏ -> chi tiết hơn nhưng feature dài hơn.
        - cell lớn -> gọn hơn nhưng mất chi tiết.
        - orientations nhiều -> hướng mịn hơn nhưng histogram thưa hơn.
        - orientations ít -> gọn hơn nhưng hướng thô hơn; orientations = 1 gần như bỏ thông tin hướng.
        - cells_per_block lớn -> chuẩn hóa vùng rộng hơn nhưng feature dài hơn.
        - preprocessing ảnh hưởng trực tiếp đến gradient đầu vào.
        """
    )
    st.warning("Phần này sẽ train lại SVM tạm thời cho cấu hình đang chọn. Cấu hình lớn như 128x128 + cell nhỏ có thể chạy lâu hơn; khi thuyết trình nên dùng visualization, chỉ bấm evaluation nếu đã chạy thử trước.")
    sample_size = st.selectbox("sample_size cho custom evaluation", ["50", "100", "300", "full"], index=0)
    if st.button("Run custom config evaluation"):
        if not (data_root / "gt.txt").exists():
            st.error("Không tìm thấy dataset/gt.txt nên không chạy custom evaluation được.")
        elif dim == 0:
            st.error("Cấu hình HOG không hợp lệ.")
        else:
            with st.spinner("Đang train SVM tạm cho custom config và đánh giá sample test..."):
                try:
                    result = custom_config_evaluation(str(data_root), custom_config, sample_size)
                    st.session_state["tab4_eval_signature"] = config_signature(custom_config)
                    st.session_state["tab4_eval_macro_f1"] = float(result["macro_f1"].iloc[0])
                    st.dataframe(format_float_df(result), width='stretch', hide_index=True)
                except Exception as exc:
                    st.error(f"Custom evaluation lỗi: {exc}")

with tab5:
    st.subheader("Kiểm tra độ bền trên điều kiện ảnh xấu")
    row = selected_row(test_metadata)
    _, roi = load_roi_from_row(row, data_root) if row is not None else (None, None)
    selected_conditions = st.multiselect("Chọn điều kiện", CONDITIONS, default=CONDITIONS if not quick_demo else CONDITIONS[:4])
    if roi is None:
        st.warning("Không có ROI để kiểm tra.")
    else:
        rows = []
        for cond in selected_conditions:
            transformed = apply_condition_transform(roi, cond)
            base_pred = predict_roi(baseline_model, transformed, baseline_config)
            opt_pred = predict_roi(optimized_model, transformed, optimized_config)
            true_label = str(row["true_label"])
            rows.append(
                {
                    "condition": cond,
                    "image": transformed,
                    "baseline_pred": base_pred,
                    "optimized_pred": opt_pred,
                    "baseline": render_prediction_badge(true_label, base_pred),
                    "optimized": render_prediction_badge(true_label, opt_pred),
                }
            )
        for item in rows:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            c1.image(item["image"], caption=item["condition"], width=130)
            c2.write(f"Baseline: **{item['baseline_pred']}**")
            c2.caption(item["baseline"])
            c3.write(f"Optimized: **{item['optimized_pred']}**")
            c3.caption(item["optimized"])
            c4.write("True label")
            c4.caption(str(row["true_label"]))

    st.write("Bảng robustness tổng hợp đã export từ notebook/export script:")
    summary_cols = ["model", "clean_macro_f1", "avg_robust_macro_f1", "worst_case_macro_f1", "drop_from_clean"]
    st.dataframe(format_float_df(robustness_summary[summary_cols]), width='stretch', hide_index=True)
    st.info("Robustness không dùng để thay đổi giả thuyết chính. Nó kiểm tra bổ sung xem optimized HOG có còn ổn khi ROI bị sáng/tối, blur hoặc noise hay không.")

with tab6:
    st.subheader("Phân tích lỗi sai")
    if per_class_f1.empty:
        st.info("Chưa có per_class_f1.csv.")
    else:
        st.dataframe(format_float_df(per_class_f1), width='stretch', hide_index=True)
        mandatory = per_class_f1[per_class_f1["class"] == "mandatory"]
        if not mandatory.empty and float(mandatory["delta_f1"].iloc[0]) == float(per_class_f1["delta_f1"].max()):
            st.markdown(
                '<div class="good-box">Trong split hiện tại, mandatory là lớp cải thiện rõ. '
                "Các ví dụ này giúp minh họa optimized HOG sửa được một số lỗi của baseline.</div>",
                unsafe_allow_html=True,
            )

    if not test_metadata.empty:
        st.markdown("**Confusion matrix tr\u00ean test set**")
        labels_for_cm = GROUP_ORDER
        baseline_cm = confusion_matrix(test_metadata["true_label"], test_metadata["baseline_pred"], labels=labels_for_cm)
        optimized_cm = confusion_matrix(test_metadata["true_label"], test_metadata["optimized_pred"], labels=labels_for_cm)
        cm_fig, cm_axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, cm, title, cmap in [
            (cm_axes[0], baseline_cm, "Baseline confusion matrix", "Blues"),
            (cm_axes[1], optimized_cm, "Optimized confusion matrix", "Greens"),
        ]:
            ax.imshow(cm, cmap=cmap)
            ax.set_title(title)
            ax.set_xticks(range(len(labels_for_cm)))
            ax.set_yticks(range(len(labels_for_cm)))
            ax.set_xticklabels(labels_for_cm, rotation=30, ha="right")
            ax.set_yticklabels(labels_for_cm)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for y_idx in range(cm.shape[0]):
                for x_idx in range(cm.shape[1]):
                    ax.text(x_idx, y_idx, int(cm[y_idx, x_idx]), ha="center", va="center", fontsize=9)
        cm_fig.tight_layout()
        st.pyplot(cm_fig, width='stretch')

    def show_gallery(df: pd.DataFrame, title: str) -> None:
        st.markdown(f"**{title}**")
        if df.empty:
            st.write("Không có ví dụ.")
            return
        gallery = df.sort_values(by="true_label", key=lambda s: (s != "mandatory")).head(8)
        cols = st.columns(4)
        for idx, (_, ex) in enumerate(gallery.iterrows()):
            with cols[idx % 4]:
                _, ex_roi = load_roi_from_row(ex, data_root)
                if ex_roi is not None:
                    st.image(ex_roi, width='stretch')
                st.caption(f"true={ex['true_label']} | B={ex['baseline_pred']} | O={ex['optimized_pred']}")
                if st.button("Mở ở tab HOG", key=f"open_{title}_{int(ex['roi_id'])}"):
                    st.session_state["selected_roi_id"] = int(ex["roi_id"])
                    st.success("Đã chọn ví dụ này. Mở Tab 3 để xem HOG visualization.")

    fixed_df = error_examples[error_examples["case_type"] == "baseline_wrong_optimized_correct"] if not error_examples.empty else pd.DataFrame()
    regressed_df = error_examples[error_examples["case_type"] == "baseline_correct_optimized_wrong"] if not error_examples.empty else pd.DataFrame()
    show_gallery(fixed_df, "Baseline sai nhưng Optimized đúng")
    show_gallery(regressed_df, "Baseline đúng nhưng Optimized sai")

with tab7:
    st.subheader("Slide mode / Thuyết trình nhanh")
    baseline_row = final_metrics[final_metrics["model"] == "Baseline HOG + SVM"].iloc[0]
    optimized_row = final_metrics[final_metrics["model"] == "Optimized HOG + SVM"].iloc[0]
    tab4_config = tab4_config_from_state()
    tab4_is_exported_optimized = configs_equal(tab4_config, optimized_config)
    tab4_eval_signature = st.session_state.get("tab4_eval_signature")
    tab4_eval_macro_f1 = st.session_state.get("tab4_eval_macro_f1")
    if tab4_is_exported_optimized:
        slide_config_label = "Optimized đã export"
        slide_macro_line = f"Macro F1 tăng từ {float(baseline_row['macro_f1']):.4f} lên {float(optimized_row['macro_f1']):.4f}."
        slide_eval_line = "Trong notebook seed 126, optimized ưu tiên cấu hình có Macro F1 cao nhất trong sweep."
    elif tab4_eval_signature == config_signature(tab4_config) and tab4_eval_macro_f1 is not None:
        slide_config_label = "Cấu hình Tab 4 đã đánh giá"
        slide_macro_line = f"Macro F1 baseline = {float(baseline_row['macro_f1']):.4f}; cấu hình Tab 4 = {float(tab4_eval_macro_f1):.4f}."
        slide_eval_line = "Macro F1 của cấu hình Tab 4 lấy từ custom evaluation vừa chạy trong demo."
    else:
        slide_config_label = "Cấu hình Tab 4 chưa đánh giá"
        slide_macro_line = f"Macro F1 baseline = {float(baseline_row['macro_f1']):.4f}; cấu hình Tab 4 chưa có Macro F1."
        slide_eval_line = "Muốn có Macro F1 cho cấu hình Tab 4 thì chạy custom evaluation ở Tab 4."
    st.markdown(
        f"""
        1. Bài toán là classification trên ROI crop, không phải detection.
        2. Baseline là HOG + SVM mặc định.
        3. HOG biểu diễn cạnh bằng histogram hướng gradient.
        4. Nhóm tối ưu/thử các tham số HOG để tìm cấu hình phù hợp hơn.
        5. Baseline: image_size={pair_label(baseline_config['image_size'])}, orientations={baseline_config['orientations']}, feature_dim={compute_feature_dim(baseline_config)}.
        6. {slide_config_label}: image_size={pair_label(tab4_config['image_size'])}, orientations={tab4_config['orientations']}, feature_dim={compute_feature_dim(tab4_config)}.
        7. {slide_macro_line}
        8. {slide_eval_line}
        9. Robustness kiểm tra mô hình dưới brightness/contrast/blur/noise.
        10. Failure analysis cho thấy optimized sửa được một số lỗi của baseline.
        """
    )
