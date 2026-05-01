# transfer_engine.py — Gửi ảnh từ Jetson → Server
import asyncio
import aiohttp
from dotenv import load_dotenv
load_dotenv()
import aiofiles
import json
import os
import time
from pathlib import Path
from loguru import logger
from typing import List
import hashlib

class TransferEngine:
    """
    Upload bất đồng bộ, có hàng đợi, tự retry
    Jetson tiếp tục chụp trong khi upload chạy nền
    """

    SERVER_URL   = os.getenv("SERVER_URL", "http://192.168.1.100:8888")
    MAX_RETRIES  = 5
    RETRY_DELAY  = 3            # giây
    MAX_PARALLEL = 4            # Upload song song

    def __init__(self, queue_dir: str = "/tmp/upload_queue"):
        self.queue_dir  = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True)
        self._semaphore = None  # tạo lazy trong async context

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.MAX_PARALLEL)
        return self._semaphore

    def enqueue(self, files: List[str], product_id: str,
                category: str, metadata: dict = None) -> str:
        """Thêm batch ảnh vào hàng đợi — gọi ngay sau session"""
        # Dùng timestamp tránh collision khi cùng sản phẩm chụp nhiều lần
        job_id   = f"{hashlib.md5(product_id.encode()).hexdigest()[:8]}_{int(time.time())}"
        job_path = self.queue_dir / f"{job_id}.json"

        job = {
            "product_id": product_id,
            "category":   category,
            "files":      files,
            "metadata":   metadata or {},
            "retries":    0,
        }
        with open(job_path, 'w') as f:
            json.dump(job, f)

        logger.info(f"[Queue] Đã thêm job {job_id}: {len(files)} ảnh")
        return job_id

    async def upload_file(self, session: aiohttp.ClientSession,
                          fpath: str, product_id: str,
                          category: str) -> bool:
        """Upload 1 file, retry ngoài semaphore để không chiếm slot khi ngủ"""
        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._get_semaphore():  # chỉ chiếm slot khi đang gửi
                    async with aiofiles.open(fpath, 'rb') as f:
                        data = await f.read()

                    form = aiohttp.FormData()
                    form.add_field('file', data,
                                   filename=os.path.basename(fpath),
                                   content_type='image/jpeg')
                    form.add_field('product_id', product_id)
                    form.add_field('category', category)

                    async with session.post(
                        f"{self.SERVER_URL}/upload",
                        data=form,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            return True
                        logger.warning(f"  Server lỗi {resp.status}, retry {attempt+1}")

            except Exception as e:
                logger.warning(f"  Upload lỗi: {e}, retry {attempt+1}/{self.MAX_RETRIES}")

            # Sleep ngoài semaphore — không giữ slot
            await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))

        logger.error(f"  FAIL sau {self.MAX_RETRIES} lần: {fpath}")
        return False

    async def process_queue(self):
        """Chạy liên tục — xử lý hàng đợi upload"""
        logger.info("[Transfer] Upload worker started")

        async with aiohttp.ClientSession() as session:
            while True:
                job_files = list(self.queue_dir.glob("*.json"))

                if not job_files:
                    await asyncio.sleep(2)
                    continue

                for job_file in job_files:
                    try:
                        with open(job_file) as f:
                            job = json.load(f)

                        product_id = job['product_id']
                        category   = job['category']
                        files      = job['files']

                        logger.info(f"[Upload] {product_id}: {len(files)} ảnh")

                        tasks   = [self.upload_file(session, fp, product_id, category)
                                   for fp in files]
                        results = await asyncio.gather(*tasks)
                        success = sum(results)
                        failed  = len(files) - success

                        if failed == 0:
                            logger.info(f"  ✓ {success}/{len(files)} ảnh thành công")
                            os.remove(job_file)  # chỉ xóa khi tất cả thành công
                            await self._notify_server(session, product_id,
                                                      category, job['metadata'])
                        else:
                            logger.warning(f"  ✗ {failed} ảnh thất bại — giữ job để retry sau")

                    except Exception as e:
                        logger.error(f"Job error: {e}")

    async def _notify_server(self, session, product_id, category, metadata):
        """Báo server biết 1 sản phẩm đã upload đủ → server tự build vector"""
        try:
            await session.post(
                f"{self.SERVER_URL}/notify_complete",
                json={
                    "product_id": product_id,
                    "category":   category,
                    "metadata":   metadata,
                },
                timeout=aiohttp.ClientTimeout(total=5)
            )
            logger.info(f"  → Server được thông báo: {product_id}")
        except Exception as e:
            logger.debug(f"  Notify thất bại (non-critical): {e}")
