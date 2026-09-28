# Phân loại hoa Iris bằng thuật toán KNN

Đồ án thực tập cơ sở ngành — **Tìm hiểu và cài đặt thuật toán KNN phân loại hoa Iris (Diên vĩ)**.

> **Sinh viên:** Lê Thanh Gia Phúc — **MSSV:** 170124926 — **Lớp:** DX24TT8
> **GVHD:** TS. Nguyễn Nhứt Lam — Trường Kỹ thuật và Công nghệ, Đại học Trà Vinh

Kho mã nguồn gồm hai cách cài đặt thuật toán K-Nearest Neighbors (KNN):

| File | Mô tả |
|------|-------|
| `knn_from_scratch.py` | Cài đặt KNN **từ đầu** (chỉ dùng `numpy`): tự tính khoảng cách, tìm K láng giềng, bình chọn đa số, tính accuracy / confusion matrix / precision / recall / F1, và chọn K bằng 5-fold cross-validation. |
| `knn_sklearn.py` | Cài đặt bằng thư viện **scikit-learn**, đồng thời vẽ và lưu các biểu đồ (phân bố dữ liệu, độ chính xác theo K, ma trận nhầm lẫn) vào thư mục `output/`. |
| `data/iris.csv` | Bộ dữ liệu Iris (150 mẫu, 3 lớp, 4 thuộc tính). |

## Bộ dữ liệu

- **150 mẫu**, chia đều **3 lớp**: `Iris-setosa`, `Iris-versicolor`, `Iris-virginica` (mỗi lớp 50 mẫu).
- **4 thuộc tính**: chiều dài/rộng đài hoa (sepal), chiều dài/rộng cánh hoa (petal), đơn vị cm.

## Cài đặt môi trường

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Chạy chương trình

```bash
# Bản tự cài đặt (không cần scikit-learn, chỉ cần numpy)
python knn_from_scratch.py

# Bản dùng scikit-learn + vẽ biểu đồ
python knn_sklearn.py
```

## Phương pháp thực nghiệm

1. Chia dữ liệu **80% huấn luyện / 20% kiểm tra**, phân tầng (stratified), `seed = 42`.
2. **Chuẩn hóa** đặc trưng bằng z-score (StandardScaler) theo thống kê tập huấn luyện.
3. **Chọn K** trong {1, 3, 5, …, 15} bằng **5-fold cross-validation**.
4. Đánh giá bằng **accuracy, confusion matrix, precision/recall/F1**.

## Kết quả tiêu biểu

- K tốt nhất theo cross-validation: **K = 7**.
- Độ chính xác trên tập kiểm tra: **≈ 96.7%** (29/30 mẫu đúng).
- Lớp `setosa` tách biệt hoàn toàn; nhầm lẫn chủ yếu giữa `versicolor` và `virginica`.

> Kết quả có thể chênh lệch nhẹ tùy phiên bản thư viện và cách chia dữ liệu.

## Cấu trúc thư mục

```
source_code/
├── data/
│   └── iris.csv
├── knn_from_scratch.py
├── knn_sklearn.py
├── requirements.txt
└── README.md
```
