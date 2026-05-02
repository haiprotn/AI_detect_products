"""
uploader.py — Chạy trên Jetson
Upload ảnh đã chụp lên Server.
"""
import asyncio
import json
import os
import time
from pathlib import Path

import aiofiles
import aiohttp
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

SERVER_URL   = os.getenv("SERVER_URL", "http://192.168.1.254:8888")
QUEUE_DIR    = Path("/tmp/upload_queue")
MAX_PARALLEL = 4
MAX_RETRIES  = 5


def enqueue(files: list, product_id: str, category: str) -> str:
    QUEUE_DIR.mkdir(exist_ok=True)
    job_id   = f"{product_id}_{int(time.time())}"
    job_path = QUEUE_DIR / f"{job_id}.json"
    json.dump({"product_id": product_id, "category": category,
               "files": files}, open(job_path, "w"))
    logger.info(f"[Queue] {job_id}: {len(files)} ảnh")
    return job_id


async def _upload_file(session, fpath, product_id, category, sem, idx, total):
    fname = Path(fpath).name
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                async with aiofiles.open(fpath, "rb") as f:
                    data = await f.read()
                form = aiohttp.FormData()
                form.add_field("file", data, filename=fname,
                               content_type="image/jpeg")
                form.add_field("product_id", product_id)
                form.add_field("category", category)
                async with session.post(f"{SERVER_URL}/upload", data=form,
                                        timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status == 200:
                        logger.info(f"  [{idx}/{total}] ✓ {fname}")
                        return True
        except Exception as e:
            logger.warning(f"  [{idx}/{total}] retry {attempt+1}: {e}")
        await asyncio.sleep(2 * (attempt + 1))
    logger.error(f"  FAIL: {fname}")
    return False


async def _notify(session, product_id, category):
    try:
        await session.post(f"{SERVER_URL}/notify_complete",
                           json={"product_id": product_id, "category": category},
                           timeout=aiohttp.ClientTimeout(total=5))
        logger.info(f"  → Server build vector: {product_id}")
    except Exception as e:
        logger.warning(f"  Notify lỗi: {e}")


async def process_queue():
    sem = asyncio.Semaphore(MAX_PARALLEL)
    async with aiohttp.ClientSession() as session:
        while True:
            jobs = list(QUEUE_DIR.glob("*.json"))
            if not jobs:
                await asyncio.sleep(2)
                continue
            for jf in jobs:
                job      = json.load(open(jf))
                pid      = job["product_id"]
                cat      = job["category"]
                files    = job["files"]
                total    = len(files)
                logger.info(f"[Upload] {pid}: {total} ảnh")
                tasks   = [_upload_file(session, f, pid, cat, sem, i+1, total)
                           for i, f in enumerate(files)]
                results = await asyncio.gather(*tasks)
                if all(results):
                    os.remove(jf)
                    await _notify(session, pid, cat)
                else:
                    logger.warning(f"  {sum(1 for r in results if not r)} ảnh thất bại")


if __name__ == "__main__":
    QUEUE_DIR.mkdir(exist_ok=True)
    # Quét ảnh chưa upload
    save_dir = Path(os.getenv("SAVE_DIR", "/home/user/data/products"))
    category = os.getenv("CATEGORY", "san_pham")
    cat_dir  = save_dir / category
    if cat_dir.exists():
        for pd in sorted(cat_dir.iterdir()):
            if pd.is_dir():
                imgs = sorted(str(f) for f in pd.glob("*.jpg"))
                if imgs:
                    enqueue(imgs, pd.name, category)

    logger.info(f"Server: {SERVER_URL}")
    logger.info("Bắt đầu upload... Ctrl+C để dừng")
    try:
        asyncio.run(process_queue())
    except KeyboardInterrupt:
        logger.info("Dừng.")
