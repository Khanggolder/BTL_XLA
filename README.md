# HOG + SVM GTSDB Demo

Dự án demo phân loại nhóm biển báo GTSDB bằng đặc trưng HOG và mô hình Linear SVM.

Demo tập trung vào giả thuyết: tối ưu cấu hình HOG có thể cải thiện Macro F1 so với baseline HOG + SVM.

## 1. Cấu trúc dữ liệu mẫu

Link dataset: https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html

Dữ liệu mẫu đặt tại:

```text
data/
└── FullIJCNN2013/
    ├── 00000.ppm
    ├── 00001.ppm
    ├── ...
    ├── 00899.ppm
    ├── gt.txt
    └── ReadMe.txt
```

Thư mục này là GTSDB - German Traffic Sign Detection Benchmark. Demo chỉ dùng ground-truth bounding box trong `gt.txt` để crop ROI biển báo, không làm detection.

## 2. Cài đặt

Nên dùng môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu dùng bash/Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Chạy demo Streamlit

Nếu thư mục `artifacts/` đã có sẵn model và kết quả export, chạy ngay:

```powershell
streamlit run app.py
```

Mở URL Streamlit hiện trong terminal, thường là:

```text
http://localhost:8501
```

## 4. Export lại artifacts khi cần

Chỉ cần chạy bước này nếu thiếu `artifacts/`, đổi dữ liệu, hoặc muốn tái tạo model/kết quả:

```powershell
python export_artifacts.py --data-root data/FullIJCNN2013 --notebook traffic_sign_cv.ipynb
```

Sau đó chạy lại app:

```powershell
streamlit run app.py
```
