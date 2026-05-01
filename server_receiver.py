# server_receiver.py — Chạy trên máy Server RTX 5060 Ti
import asyncio
import os
import re
from pathlib import Path
from datetime import datetime

import aiofiles
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

app = FastAPI(title="Product Image Server")

DATA_DIR      = Path("./received_images")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def safe_filename(name: str) -> str:
    """Loại bỏ ký tự nguy hiểm, chỉ giữ tên file — không cho path traversal."""
    name = os.path.basename(name)               # bỏ thư mục
    name = re.sub(r"[^\w.\-]", "_", name)       # chỉ giữ chữ/số/._-
    return name or "unnamed.jpg"


@app.post("/upload")
async def receive_image(
    file:       UploadFile = File(...),
    product_id: str = Form(...),
    category:   str = Form(...),
):
    # Kiểm tra kích thước
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413,
                            detail=f"File quá lớn (max {MAX_FILE_SIZE//1024//1024}MB)")

    # Sanitize tên file — tránh path traversal
    filename  = safe_filename(file.filename or "image.jpg")
    save_dir  = DATA_DIR / safe_filename(category) / safe_filename(product_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    fpath = save_dir / filename
    async with aiofiles.open(fpath, 'wb') as f:
        await f.write(content)

    return JSONResponse({"status": "ok", "saved": str(fpath)})


class NotifyPayload(BaseModel):
    product_id: str
    category:   str
    metadata:   dict = {}


@app.post("/notify_complete")
async def product_complete(
    background_tasks: BackgroundTasks,
    data: NotifyPayload,
):
    """Sau khi nhận đủ ảnh, tự động build vector"""
    logger.info(f"[Complete] {data.product_id} → Queue build vector")
    background_tasks.add_task(
        build_vector_for_product,
        data.product_id, data.category, data.metadata
    )
    return JSONResponse({"status": "queued"})


async def build_vector_for_product(product_id: str,
                                    category: str, metadata: dict):
    """Tự động tạo vector sau khi nhận xong ảnh — non-blocking."""
    logger.info(f"[BuildVector] Bắt đầu: {product_id}")
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "build_single_product_vector.py",
            "--product_id", product_id,
            "--category",   category,
            "--data_dir",   str(DATA_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode == 0:
            logger.success(f"[BuildVector] Xong: {product_id}")
        else:
            logger.error(f"[BuildVector] Lỗi: {stderr.decode()}")
    except Exception as e:
        logger.error(f"[BuildVector] Exception: {e}")


@app.get("/stats")
async def get_stats():
    """Dashboard đơn giản"""
    if not DATA_DIR.exists():
        return {}

    stats = {}
    for cat_dir in DATA_DIR.iterdir():
        if cat_dir.is_dir():
            products = [p for p in cat_dir.iterdir() if p.is_dir()]
            stats[cat_dir.name] = {
                "products": len(products),
                "images":   sum(len(list(p.glob("*.jpg"))) for p in products),
            }
    return stats


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8888, workers=4)
