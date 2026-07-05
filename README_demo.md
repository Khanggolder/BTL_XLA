# Demo Streamlit: HOG + SVM trên GTSDB

Demo này phục vụ thuyết trình cho giả thuyết:

> Tối ưu cấu hình HOG giúp cải thiện Macro F1 so với baseline HOG + SVM mặc định.

Phạm vi demo là classification trên ROI biển báo đã crop từ ground-truth bounding box hoặc ảnh ROI người dùng upload. Demo không làm detection bbox, không sinh proposal, không dùng CNN/YOLO, không thêm Color + HOG hay voting.

## Cách chạy

1. Đặt GTSDB tại:

```text
data/FullIJCNN2013/
```

Thư mục này cần có `gt.txt` và các ảnh `.ppm`.

2. Cài thư viện:

```bash
pip install -r requirements.txt
```

3. Export artifacts:

```bash
python export_artifacts.py --data-root data/FullIJCNN2013 --notebook traffic_sign_cv.ipynb
```

Script sẽ lưu:

```text
artifacts/baseline_model.joblib
artifacts/optimized_model.joblib
artifacts/configs.json
artifacts/test_metadata.csv
artifacts/final_metrics.csv
artifacts/robustness_results.csv
artifacts/robustness_summary.csv
artifacts/error_examples.csv
artifacts/per_class_f1.csv
```

4. Chạy Streamlit:

```bash
streamlit run streamlit_app.py
```

## Cấu hình dữ liệu

Mặc định demo đọc dữ liệu từ `data/FullIJCNN2013/`. Nếu cần đổi đường dẫn:

```bash
$env:DATA_ROOT="duong/dan/FullIJCNN2013"
python export_artifacts.py --data-root data/FullIJCNN2013 --notebook traffic_sign_cv.ipynb
streamlit run streamlit_app.py
```

Trong app cũng có ô `DATA_ROOT` ở sidebar để đổi đường dẫn ảnh khi cần.

## Nội dung chính

- Baseline HOG + SVM: `image_size=(64, 64)`, `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `preprocessing=gray_clahe`.
- Optimized HOG + SVM: doc tu `artifacts/configs.json`; voi notebook seed 126 hien tai thuong la `image_size=(96, 96)`, `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `preprocessing=gray_clahe`.
- App chỉ load model đã export, không train lại khi mở.
- HOG visualization chỉ tính cho ROI đang chọn.
- Custom HOG evaluation chỉ chạy khi bấm `Run custom config evaluation`.
