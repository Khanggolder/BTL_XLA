# Traffic Sign CV Final Project

## Dataset

Tải GTSDB - German Traffic Sign Detection Benchmark, sau đó giải nén theo cấu trúc:

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

Khi chạy notebook, nếu thiếu `data/FullIJCNN2013/gt.txt` hoặc ảnh `.ppm`, notebook sẽ dừng và in hướng dẫn đặt dataset đúng vị trí. Notebook không tạo dataset giả và không hard-code kết quả.

## Cài thư viện

```bash
pip install numpy pandas opencv-python scikit-image scikit-learn matplotlib tqdm
```

## Chạy notebook

Mở `traffic_sign_cv.ipynb` trong Jupyter Notebook, JupyterLab hoặc VS Code, sau đó chạy lần lượt từ đầu đến cuối.
