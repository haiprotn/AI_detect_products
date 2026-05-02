"""
server.py — Chạy trên Server RTX 5060 Ti
Nhận ảnh từ Jetson → lưu → tự build vector khi đủ ảnh.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

import aiofiles
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

app = FastAPI(title="AI Detect Product Server")

DATA_DIR      = Path(os.getenv("DATA_DIR", "./received_images"))
VECTOR_DIR    = Path(os.getenv("VECTOR_DIR", "./vector_store"))
MAX_FILE_SIZE = 10 * 1024 * 1024


def safe_name(s: str) -> str:
    return re.sub(r"[^\w.\-]", "_", os.path.basename(s)) or "unnamed"


@app.post("/upload")
async def upload(
    file:       UploadFile = File(...),
    product_id: str = Form(...),
    category:   str = Form(...),
):
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, "File quá lớn")

    save_dir = DATA_DIR / safe_name(category) / safe_name(product_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    fpath = save_dir / safe_name(file.filename or "img.jpg")

    async with aiofiles.open(fpath, "wb") as f:
        await f.write(data)

    logger.info(f"[Upload] {product_id}/{fpath.name} ({len(data)//1024}KB)")
    return JSONResponse({"status": "ok", "path": str(fpath)})


class NotifyPayload(BaseModel):
    product_id: str
    category:   str


@app.post("/notify_complete")
async def notify_complete(bg: BackgroundTasks, data: NotifyPayload):
    logger.info(f"[Notify] {data.product_id} → build vector")
    bg.add_task(_build_vectors, data.product_id, data.category)
    return JSONResponse({"status": "queued"})


async def _build_vectors(product_id: str, category: str):
    script = Path(__file__).parent / "build_vectors.py"
    cmd = [sys.executable, str(script),
           "--product_id", product_id,
           "--category", category,
           "--data_dir", str(DATA_DIR),
           "--vector_dir", str(VECTOR_DIR)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.success(f"[Build] Xong: {product_id}")
        else:
            logger.error(f"[Build] Lỗi: {stderr.decode(errors='replace')}")
    except Exception as e:
        logger.error(f"[Build] Exception: {e}")


@app.get("/stats")
async def stats():
    result = {}
    for cat in DATA_DIR.iterdir():
        if not cat.is_dir():
            continue
        products = [p for p in cat.iterdir() if p.is_dir()]
        result[cat.name] = {
            "products": len(products),
            "images": sum(len(list(p.glob("*.jpg"))) for p in products),
        }
    vecs = list(VECTOR_DIR.glob("*.pkl")) if VECTOR_DIR.exists() else []
    result["_vectors"] = len(vecs)
    return result


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    VECTOR_DIR.mkdir(exist_ok=True)
    uvicorn.run("server:app", host="0.0.0.0", port=8888, workers=1)
