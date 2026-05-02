"""
build_vectors.py — Chạy trên Server RTX 5060 Ti
Build DINOv2 vectors từ ảnh sản phẩm, lưu ra .pkl
"""
import argparse
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from loguru import logger

warnings.filterwarnings("ignore")

MODEL_NAME = "dinov2_vitb14"   # base model — 768 dim, chính xác hơn vits14
EMBED_DIM  = 768
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_model = None


def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading {MODEL_NAME} on {DEVICE}...")
        _model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        _model.eval().to(DEVICE)
    return _model


def embed(img_path: Path) -> np.ndarray | None:
    try:
        img = Image.open(img_path).convert("RGB")
        t   = _transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vec = get_model()(t).cpu().numpy().flatten()
        vec = vec / np.linalg.norm(vec)
        return vec.astype(np.float32)
    except Exception as e:
        logger.warning(f"  Skip {img_path.name}: {e}")
        return None


def build(product_id: str, category: str,
          data_dir: Path, vector_dir: Path) -> bool:

    img_dir = data_dir / category / product_id
    imgs    = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    if not imgs:
        logger.error(f"Không có ảnh trong {img_dir}")
        return False

    logger.info(f"[{product_id}] {len(imgs)} ảnh → embed...")
    vecs = [v for p in imgs if (v := embed(p)) is not None]

    if not vecs:
        logger.error("Không embed được ảnh nào")
        return False

    vector_dir.mkdir(exist_ok=True)
    out = vector_dir / f"{product_id}.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "product_id": product_id,
            "category":   category,
            "vectors":    np.stack(vecs),
            "n_images":   len(vecs),
            "embed_dim":  EMBED_DIM,
            "model":      MODEL_NAME,
            "created_at": datetime.now().isoformat(),
        }, f)

    logger.success(f"  {product_id}: {len(vecs)} vectors → {out.name}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product_id",  required=True)
    parser.add_argument("--category",    required=True)
    parser.add_argument("--data_dir",    default="./received_images")
    parser.add_argument("--vector_dir",  default="./vector_store")
    args = parser.parse_args()

    ok = build(args.product_id, args.category,
               Path(args.data_dir), Path(args.vector_dir))
    sys.exit(0 if ok else 1)
