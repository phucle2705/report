"""
knn_from_scratch.py
-------------------
Cài đặt thuật toán K-Nearest Neighbors (KNN) TỪ ĐẦU (không dùng thư viện ML)
để phân loại hoa Iris.

Chỉ dùng numpy cho tính toán ma trận. Toàn bộ logic KNN (tính khoảng cách,
tìm K láng giềng, bình chọn đa số) đều tự viết.

Chạy:
    python knn_from_scratch.py
"""

import csv
import os
from collections import Counter

import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "iris.csv")
SEED = 42


# ----------------------------------------------------------------------
# 1. Đọc dữ liệu
# ----------------------------------------------------------------------
def load_iris_csv(path):
    X, y, names = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # bỏ dòng tiêu đề
        for row in reader:
            if not row:
                continue
            X.append([float(v) for v in row[:4]])
            label = row[4]
            if label not in names:
                names.append(label)
            y.append(names.index(label))
    return np.array(X), np.array(y), names


# ----------------------------------------------------------------------
# 2. Chia tập train/test có phân tầng (stratified)
# ----------------------------------------------------------------------
def stratified_split(X, y, test_ratio=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = round(len(idx) * test_ratio)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx])


# ----------------------------------------------------------------------
# 3. Chuẩn hóa (z-score) theo thống kê của tập huấn luyện
# ----------------------------------------------------------------------
class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------
# 4. Thuật toán KNN tự cài đặt
# ----------------------------------------------------------------------
class KNNClassifier:
    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _distance(self, a, b):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2, axis=1))
        elif self.metric == "manhattan":
            return np.sum(np.abs(a - b), axis=1)
        raise ValueError("metric không hợp lệ")

    def predict_one(self, x):
        dists = self._distance(self.X_train, x)          # khoảng cách tới mọi điểm
        nn_idx = np.argsort(dists)[: self.k]             # chỉ số K điểm gần nhất
        nn_labels = self.y_train[nn_idx]                 # nhãn của K láng giềng
        return Counter(nn_labels).most_common(1)[0][0]   # bình chọn đa số

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# ----------------------------------------------------------------------
# 5. Các độ đo đánh giá
# ----------------------------------------------------------------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def classification_report(cm, names):
    print(f"{'Lớp':<16}{'Precision':>10}{'Recall':>9}{'F1':>8}{'Support':>9}")
    for c, name in enumerate(names):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"{name:<16}{prec:>10.3f}{rec:>9.3f}{f1:>8.3f}{cm[c].sum():>9}")


# ----------------------------------------------------------------------
# 6. Chọn K bằng kiểm định chéo 5-fold có phân tầng
# ----------------------------------------------------------------------
def cross_val_choose_k(X, y, k_values, folds=5, seed=SEED):
    rng = np.random.default_rng(seed + 7)
    # tạo các fold phân tầng
    fold_idx = [[] for _ in range(folds)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            fold_idx[i % folds].append(ix)

    results = {}
    for k in k_values:
        accs = []
        for f in range(folds):
            test_i = np.array(fold_idx[f])
            train_i = np.array([i for j in range(folds) if j != f for i in fold_idx[j]])
            sc = StandardScaler().fit(X[train_i])
            clf = KNNClassifier(k=k).fit(sc.transform(X[train_i]), y[train_i])
            accs.append(accuracy(y[test_i], clf.predict(sc.transform(X[test_i]))))
        results[k] = (np.mean(accs), np.std(accs))
    return results


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    X, y, names = load_iris_csv(DATA_PATH)
    print(f"Tổng số mẫu: {len(X)} | Số lớp: {len(names)} | Số thuộc tính: {X.shape[1]}")

    X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2)
    print(f"Tập huấn luyện: {len(X_train)} | Tập kiểm tra: {len(X_test)}\n")

    sc = StandardScaler().fit(X_train)
    X_train_s, X_test_s = sc.transform(X_train), sc.transform(X_test)

    # --- chọn K bằng cross-validation ---
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    cv = cross_val_choose_k(X, y, k_values)
    print("Kết quả 5-fold cross-validation:")
    for k in k_values:
        print(f"  K = {k:>2}:  {cv[k][0]*100:.2f}%  (± {cv[k][1]*100:.2f}%)")
    best_k = max(k_values, key=lambda k: (cv[k][0], -k))
    print(f"\n=> K tốt nhất theo CV: {best_k}\n")

    # --- đánh giá trên tập kiểm tra ---
    clf = KNNClassifier(k=best_k).fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = accuracy(y_test, y_pred)
    print(f"Độ chính xác trên tập kiểm tra (K={best_k}): {acc*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred, len(names))
    print("\nMa trận nhầm lẫn:")
    print(cm)
    print()
    classification_report(cm, names)

    # --- thử với một mẫu mới ---
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # một bông hoa mới cần phân loại
    pred = clf.predict(sc.transform(sample))[0]
    print(f"\nDự đoán mẫu {sample[0].tolist()} -> {names[pred]}")


if __name__ == "__main__":
    main()
