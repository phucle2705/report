"""
knn_sklearn.py
--------------
Cài đặt KNN phân loại hoa Iris dùng thư viện scikit-learn, đồng thời
vẽ và lưu các biểu đồ minh họa (scatter, accuracy theo K, ma trận nhầm lẫn).

Chạy:
    python knn_sklearn.py

Kết quả và hình ảnh sẽ được lưu trong thư mục ./output
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # không cần màn hình, chỉ lưu file
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

SEED = 42
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    # 1. Nạp dữ liệu
    iris = load_iris()
    X, y = iris.data, iris.target
    names = list(iris.target_names)
    print(f"Dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} thuộc tính, {len(names)} lớp {names}")

    # 2. Chia train/test (phân tầng)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3. Chuẩn hóa
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. Chọn K bằng 5-fold cross-validation
    k_values = list(range(1, 16, 2))
    cv_means, cv_stds = [], []
    for k in k_values:
        scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=k),
            StandardScaler().fit_transform(X), y, cv=5,
        )
        cv_means.append(scores.mean())
        cv_stds.append(scores.std())
        print(f"K = {k:>2}:  CV = {scores.mean()*100:.2f}% (± {scores.std()*100:.2f}%)")
    best_k = k_values[int(np.argmax(cv_means))]
    print(f"\n=> K tốt nhất: {best_k}")

    # 5. Huấn luyện & đánh giá
    clf = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nĐộ chính xác trên tập kiểm tra (K={best_k}): {acc*100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=names))

    # 6a. Biểu đồ phân bố (petal length vs petal width)
    plt.figure(figsize=(8, 5))
    for c, name in enumerate(names):
        m = y == c
        plt.scatter(X[m, 2], X[m, 3], label=name, alpha=0.7)
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.title("Phan bo du lieu Iris")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "scatter.png"), dpi=120)
    plt.close()

    # 6b. Accuracy theo K (test + CV)
    test_acc = []
    for k in k_values:
        c = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
        test_acc.append(accuracy_score(y_test, c.predict(X_test_s)))
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, test_acc, "o-", label="Test")
    plt.plot(k_values, cv_means, "s--", label="5-fold CV")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.title("Do chinh xac theo K")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "accuracy_vs_k.png"), dpi=120)
    plt.close()

    # 6c. Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion matrix (K={best_k})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120)
    plt.close()

    print(f"Đã lưu các biểu đồ vào: {OUT}")


if __name__ == "__main__":
    main()
