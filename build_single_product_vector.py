#!/usr/bin/env python3
"""
build_single_product_vector.py
Dùng DINOv2 + FAISS — phân biệt sản phẩm tương tự tốt hơn ResNet50
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision import transforms

VECTOR_STORE_DIR = Path("./vector_store")
VECTOR_STORE_DIR.mkdir(exist_ok=True)

EMBED_DIM  = 768   # DINOv2-small output dim
MODEL_NAME = "dinov2_vits14"


class VectorBuilder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")
        self._load_model()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self):
        logger.info(f"Loading {MODEL_NAME}...")
        self.model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model ready")

    def extract(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                vec = self.model(tensor)
            vec = vec.cpu().numpy().flatten()
            vec = vec / np.linalg.norm(vec)  # L2 normalize
            return vec.astype(np.float32)
        except Exception as e:
            logger.warning(f"Lỗi {image_path.name}: {e}")
            return None

    def build(self, image_paths: List[Path], product_id: str, category: str):
        """Lưu TẤT CẢ vectors riêng lẻ — không average."""
        vectors = []
        for p in image_paths:
            v = self.extract(p)
            if v is not None:
                vectors.append(v)
            logger.debug(f"  {p.name}: {'OK' if v is not None else 'SKIP'}")

        if not vectors:
            logger.error("Không có vector nào được trích xuất")
            return False

        mat = np.stack(vectors)  # shape: (N, 768)

        data = {
            "product_id":  product_id,
            "category":    category,
            "vectors":     mat,        # lưu hết N vectors
            "n_images":    len(vectors),
            "embed_dim":   EMBED_DIM,
            "model":       MODEL_NAME,
            "created_at":  datetime.now().isoformat(),
        }
        out = VECTOR_STORE_DIR / f"{product_id}.pkl"
        with open(out, "wb") as f:
            pickle.dump(data, f)

        logger.success(f"Lưu {len(vectors)} vectors → {out}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product_id", required=True)
    parser.add_argument("--category",   required=True)
    parser.add_argument("--data_dir",   required=True)
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    product_dir = data_dir / args.category / args.product_id

    if not product_dir.exists():
        logger.error(f"Không tìm thấy: {product_dir}")
        sys.exit(1)

    images = sorted(product_dir.glob("*.jpg")) + sorted(product_dir.glob("*.png"))
    logger.info(f"Tìm thấy {len(images)} ảnh cho {args.product_id}")

    builder = VectorBuilder()
    ok = builder.build(images, args.product_id, args.category)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
