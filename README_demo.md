# README Demo Streamlit

Ứng dụng Streamlit trong `app.py` dùng để thuyết trình bài toán phân loại nhóm biển báo GTSDB bằng HOG + Linear SVM.

Demo tập trung vào giả thuyết: tối ưu cấu hình HOG có thể cải thiện Macro F1 so với baseline HOG + SVM.

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
