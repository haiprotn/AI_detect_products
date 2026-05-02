# server_receiver.py — Chạy trên máy Server RTX 5060 Ti
import asyncio
import io
import json
import os
import re
import sys
from pathlib import Path

import aiofiles
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Product Image Server")

DATA_DIR      = Path("./received_images")
VECTOR_SCRIPT = Path(__file__).parent / "build_single_product_vector.py"
VECTOR_STORE  = Path("./vector_store")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
EMBED_DIM     = 128
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K         = 5

# ── Singleton model + FAISS index (load 1 lần khi khởi động) ─────────────────

_model       = None
_faiss_index = None
_meta        = None
_transform   = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def _get_model():
    global _model
    if _model is None:
        try:
            import timm
            backbone = timm.create_model("mobilenetv3_small_100",
                                         pretrained=True, num_classes=0)
        except ImportError:
            import torchvision.models as M
            backbone = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.DEFAULT)
            backbone.classifier = torch.nn.Identity()

        backbone.eval()
        with torch.no_grad():
            dummy       = torch.zeros(1, 3, 224, 224)
            in_features = backbone(dummy).shape[1]

        _model = torch.nn.Sequential(
            backbone,
            torch.nn.Linear(in_features, EMBED_DIM),
        ).eval().to(DEVICE)
        logger.info(f"Model loaded ({in_features}→{EMBED_DIM}d) | device={DEVICE}")
    return _model


def _get_index():
    global _faiss_index, _meta
    import faiss
    index_path = VECTOR_STORE / "products.index"
    meta_path  = VECTOR_STORE / "metadata.json"
    if not index_path.exists():
        return None, []
    _faiss_index = faiss.read_index(str(index_path))
    _meta        = json.loads(meta_path.read_text(encoding="utf-8"))
    return _faiss_index, _meta


def _reload_index():
    """Reload FAISS sau khi build xong."""
    global _faiss_index, _meta
    _faiss_index, _meta = None, None
    return _get_index()


def _embed_image(img_bytes: bytes) -> np.ndarray:
    model = _get_model()
    img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    t     = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vec = model(t)
        vec = F.normalize(vec, dim=1)
    return vec.cpu().numpy().astype("float32")


def safe_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "unnamed.jpg"


# ── Upload ảnh ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def receive_image(
    file:       UploadFile = File(...),
    product_id: str = Form(...),
    category:   str = Form(...),
):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File quá lớn (max {MAX_FILE_SIZE // 1024 // 1024}MB)"
        )

    filename = safe_filename(file.filename or "image.jpg")
    save_dir = DATA_DIR / safe_filename(category) / safe_filename(product_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    fpath = save_dir / filename
    async with aiofiles.open(fpath, "wb") as f:
        await f.write(content)

    logger.info(f"[Upload] {product_id}/{filename} ({len(content)//1024}KB)")
    return JSONResponse({"status": "ok", "saved": str(fpath)})


# ── Notify hoàn tất → build vector + export RV1106 ───────────────────────────

class NotifyPayload(BaseModel):
    product_id:    str
    category:      str
    metadata:      dict = {}
    export_rv1106: bool = True   # mặc định luôn export sau khi build


@app.post("/notify_complete")
async def product_complete(
    background_tasks: BackgroundTasks,
    data: NotifyPayload,
):
    logger.info(f"[Complete] {data.product_id} → Queue build vector")
    background_tasks.add_task(
        _build_and_export,
        data.product_id, data.category, data.export_rv1106,
    )
    return JSONResponse({"status": "queued", "product_id": data.product_id})


async def _build_and_export(product_id: str, category: str, export_rv1106: bool):
    """Build FAISS vector, sau đó export gói int8 cho RV1106 (non-blocking)."""
    logger.info(f"[BuildVector] Bắt đầu: {product_id}")

    cmd = [
        sys.executable, str(VECTOR_SCRIPT),
        "--product_id", product_id,
        "--category",   category,
        "--data_dir",   str(DATA_DIR),
    ]
    if export_rv1106:
        cmd.append("--export_rv1106")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode == 0:
            logger.success(f"[BuildVector] Xong: {product_id}")
            if export_rv1106:
                logger.success(f"[Export RV1106] Gói int8 đã sẵn sàng: {product_id}")
        else:
            logger.error(
                f"[BuildVector] Lỗi [{product_id}]:\n{stderr.decode(errors='replace')}"
            )
        # Reload FAISS index để recognize ngay lập tức
        if proc.returncode == 0:
            _reload_index()
    except FileNotFoundError:
        logger.error(f"[BuildVector] Không tìm thấy script: {VECTOR_SCRIPT}")
    except Exception as e:
        logger.error(f"[BuildVector] Exception [{product_id}]: {e}")


# ── Nhận dạng sản phẩm ───────────────────────────────────────────────────────

@app.post("/recognize")
async def recognize_product(
    file:  UploadFile = File(...),
    top_k: int = TOP_K,
):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File quá lớn")

    index, meta = _get_index()
    if index is None or index.ntotal == 0:
        raise HTTPException(status_code=503,
                            detail="Chưa có vector nào — hãy build trước")

    vec = _embed_image(content)
    k   = min(top_k, index.ntotal)
    scores, ids = index.search(vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        m = meta[idx]
        results.append({
            "product_id": m["product_id"],
            "category":   m["category"],
            "score":      round(float(score), 4),   # cosine similarity
            "image":      m.get("image", ""),
        })

    # Gộp theo product_id, lấy score trung bình
    agg: dict[str, dict] = {}
    for r in results:
        pid = r["product_id"]
        if pid not in agg:
            agg[pid] = {"product_id": pid, "category": r["category"],
                        "score": 0.0, "votes": 0}
        agg[pid]["score"]  += r["score"]
        agg[pid]["votes"]  += 1

    ranked = sorted(agg.values(),
                    key=lambda x: x["score"] / x["votes"],
                    reverse=True)
    for r in ranked:
        r["score"] = round(r["score"] / r["votes"], 4)
        del r["votes"]

    logger.info(f"[Recognize] Top1: {ranked[0]['product_id']} "
                f"score={ranked[0]['score']}" if ranked else "[Recognize] Không khớp")
    return JSONResponse({"results": ranked})


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    if not DATA_DIR.exists():
        return {}

    stats = {}
    for cat_dir in DATA_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        products = [p for p in cat_dir.iterdir() if p.is_dir()]
        stats[cat_dir.name] = {
            "products": len(products),
            "images":   sum(
                len(list(p.glob("*.jpg")) + list(p.glob("*.png")))
                for p in products
            ),
        }
    return stats


@app.get("/vector_stats")
async def get_vector_stats():
    """Thống kê FAISS index và gói RV1106."""
    vector_store = Path("./vector_store")
    result: dict = {"faiss": {}, "rv1106": {}}

    index_file = vector_store / "products.index"
    meta_file  = vector_store / "metadata.json"
    if index_file.exists() and meta_file.exists():
        import json
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        products = {m["product_id"] for m in meta}
        result["faiss"] = {
            "total_vectors": len(meta),
            "total_products": len(products),
            "index_size_kb": round(index_file.stat().st_size / 1024, 1),
        }

    rv_dir = vector_store / "rv1106"
    int8_file = rv_dir / "vectors_int8.npy"
    if int8_file.exists():
        result["rv1106"] = {
            "size_kb": round(int8_file.stat().st_size / 1024, 1),
            "ready":   True,
        }

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    uvicorn.run("server_receiver:app", host="0.0.0.0", port=8888, workers=4)
