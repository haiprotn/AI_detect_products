#!/usr/bin/env python3
"""
check_similarity.py — So sánh sản phẩm dùng voting từ top-K neighbors
"""

import pickle
from pathlib import Path
import numpy as np

VECTOR_STORE_DIR = Path("./vector_store")
TOP_K = 5  # Lấy K vector gần nhất để vote


def load(product_id: str) -> dict:
    path = VECTOR_STORE_DIR / f"{product_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def best_match_score(query_mat: np.ndarray, db_mat: np.ndarray) -> float:
    """
    Voting: mỗi query vector tìm K vector gần nhất trong db.
    Score = trung bình của top-K similarity cao nhất.
    Tốt hơn average vector vì giữ được đặc trưng từng góc.
    """
    # query_mat: (Nq, D), db_mat: (Nd, D) — đã normalize
    sims = query_mat @ db_mat.T          # (Nq, Nd)
    top_k = np.sort(sims, axis=1)[:, -TOP_K:]  # top-K mỗi hàng
    return float(top_k.mean())


def main():
    pkls = list(VECTOR_STORE_DIR.glob("*.pkl"))
    if not pkls:
        print("Chưa có vector nào trong vector_store/")
        return

    products = {}
    for p in pkls:
        d = load(p.stem)
        products[p.stem] = d
        print(f"  {p.stem}: {d['n_images']} ảnh, model={d['model']}")

    ids = list(products.keys())
    W = max(len(i) for i in ids) + 2

    print(f"\nSimilarity Matrix (top-{TOP_K} voting):")
    print(" " * W, end="")
    for i in ids:
        print(f"{i[:12]:>13}", end="")
    print()

    for i in ids:
        print(f"{i:{W}}", end="")
        for j in ids:
            score = best_match_score(products[i]["vectors"],
                                     products[j]["vectors"])
            print(f"{score:.4f}       ", end="")
        print()

    # Tìm ngưỡng phân biệt
    print("\nPhân tích ngưỡng:")
    same, diff = [], []
    for i in ids:
        for j in ids:
            s = best_match_score(products[i]["vectors"], products[j]["vectors"])
            if i == j:
                same.append(s)
            else:
                diff.append(s)

    print(f"  Same product  (min): {min(same):.4f}")
    print(f"  Diff products (max): {max(diff):.4f}")
    gap = min(same) - max(diff)
    print(f"  Gap:                 {gap:.4f}  {'✓ Phân biệt được' if gap > 0.01 else '✗ Quá gần — cần thêm dữ liệu'}")


if __name__ == "__main__":
    main()
