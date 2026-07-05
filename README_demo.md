# README Demo Streamlit

Ứng dụng Streamlit trong `app.py` dùng để thuyết trình bài toán phân loại nhóm biển báo GTSDB bằng HOG + Linear SVM.

Demo tập trung vào giả thuyết: tối ưu cấu hình HOG có thể cải thiện Macro F1 so với baseline HOG + SVM.

## Phạm vi demo

Demo làm:

- crop ROI biển báo từ ground-truth bbox trong `gt.txt`;
- trích xuất HOG từ ROI sau preprocessing;
- phân loại ROI vào 4 nhóm: `prohibitory`, `danger`, `mandatory`, `other`;
- so sánh baseline HOG + SVM với cấu hình HOG tối ưu/export hoặc cấu hình đang thử ở Tab 4.

Demo không làm:

- detection bounding box;
- candidate proposal;
- CNN/YOLO;
- upload ROI ngoài dataset;
- voting hoặc Color + HOG.

## Cài đặt nhanh

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy trên dữ liệu mẫu

Dữ liệu mẫu nằm tại:

```text
data/FullIJCNN2013/
```

Thư mục này cần có:

```text
gt.txt
00000.ppm
00001.ppm
...
```

Nếu `artifacts/` đã tồn tại, chạy app:

```powershell
streamlit run app.py
```

Nếu thiếu artifacts hoặc muốn tạo lại từ dữ liệu mẫu:

```powershell
python export_artifacts.py --data-root data/FullIJCNN2013 --notebook traffic_sign_cv.ipynb
streamlit run app.py
```

## Artifacts cần có

```text
artifacts/
  baseline_model.joblib
  optimized_model.joblib
  configs.json
  test_metadata.csv
  final_metrics.csv
  robustness_results.csv
  robustness_summary.csv
  error_examples.csv
  per_class_f1.csv
```

Nếu app báo thiếu file trong danh sách này, hãy chạy lại `export_artifacts.py`.

## Ý nghĩa các tab

### 1. Tổng quan

Tóm tắt input, output, pipeline xử lý và giả thuyết chính. Bảng tham số có 2 dòng:

- Baseline giữ nguyên theo cấu hình export.
- Optimized lấy theo cấu hình đang chọn ở Tab 4.

Nếu cấu hình Tab 4 chưa được evaluation, `macro_f1` sẽ hiện `chưa đánh giá`.

### 2. Phân loại ROI

Chọn ROI từ test set và xem dự đoán của baseline/optimized model đã export. Selectbox `Chọn ví dụ thuyết trình` hỗ trợ chọn nhanh các nhóm case như baseline sai nhưng optimized đúng, cả hai đúng, cả hai sai hoặc tất cả mẫu.

### 3. Trực quan HOG

Hiển thị ROI, preprocessing, gradient magnitude/orientation, HOG baseline, HOG theo cấu hình Tab 4 và toàn bộ vector HOG đưa vào SVM. Phần histogram cho phép đổi `pixels_per_cell`; histogram optimized dùng preprocessing theo cấu hình Tab 4.

### 4. Thử tham số

Thử cấu hình HOG tương tác:

- `image_size`: `32x32`, `48x48`, `64x64`, `80x80`, `96x96`, `128x128`
- `orientations`: slider `1..18`
- `pixels_per_cell`: `2x2`, `4x4`, `8x8`, `12x12`, `16x16`
- `cells_per_block`: `1x1`, `2x2`, `3x3`, `4x4`
- `preprocessing`: các kiểu tiền xử lý đang hỗ trợ

Bấm `Cập nhật visualization và bảng Tab 1` để cập nhật ảnh HOG, feature_dim và bảng Tab 1.

Bấm `Run custom config evaluation` khi muốn train/evaluate tạm cấu hình đang chọn. Nên dùng `sample_size=50` hoặc `100` khi demo nhanh.

### 5. Robustness

Kiểm tra dự đoán trên ROI bị biến đổi sáng/tối, blur, noise. Nút `Demo nhanh` trong sidebar chỉ chọn sẵn ít điều kiện hơn để phần này gọn khi thuyết trình.

### 6. Lỗi sai

Xem per-class F1, confusion matrix và ví dụ lỗi sai/sửa lỗi. Phần này dùng metadata đã export, không đổi theo Tab 4.

### 7. Case Study Pipeline

Gom một ví dụ ROI vào một màn hình thuyết trình: ảnh gốc có bbox, ROI crop, preprocessing, gradient magnitude/orientation, HOG visualization, dự đoán baseline/optimized, parameter sweep presets và decision score của SVM.

Phần visualization và parameter sweep dùng cấu hình Tab 4. Phần prediction và SVM decision score vẫn dùng model/config optimized đã export để tránh sai kích thước vector khi cấu hình Tab 4 khác model đã train.

## Lỗi thường gặp

### Không tìm thấy dữ liệu

Kiểm tra sidebar `DATA_ROOT`. Mặc định là:

```text
data/FullIJCNN2013
```

Thư mục này phải có `gt.txt` và ảnh `.ppm`.

### Không tìm thấy artifacts

Chạy:

```powershell
python export_artifacts.py --data-root data/FullIJCNN2013 --notebook traffic_sign_cv.ipynb
```

### Evaluation quá lâu

Giảm `sample_size` ở Tab 4 xuống `50` hoặc `100`. Tránh cấu hình quá nặng như `128x128` với `pixels_per_cell=2x2` khi đang thuyết trình.