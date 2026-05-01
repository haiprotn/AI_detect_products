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

    SERVER_URL   = os.getenv("SERVER_URL", "http://192.168.1.254:8888")
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
                          category: str, index: int = 0, total: int = 0) -> bool:
        """Upload 1 file, in tiến trình, retry ngoài semaphore"""
        fname = os.path.basename(fpath)
        for attempt in range(self.MAX_RETRIES):
            try:
                async with self._get_semaphore():
                    async with aiofiles.open(fpath, 'rb') as f:
                        data = await f.read()

                    size_kb = len(data) / 1024
                    form = aiohttp.FormData()
                    form.add_field('file', data,
                                   filename=fname,
                                   content_type='image/jpeg')
                    form.add_field('product_id', product_id)
                    form.add_field('category', category)

                    async with session.post(
                        f"{self.SERVER_URL}/upload",
                        data=form,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f"  [{index:02d}/{total}] ✓ {fname} ({size_kb:.0f}KB)")
                            return True
                        logger.warning(f"  [{index:02d}/{total}] ✗ {fname} — server {resp.status}, retry {attempt+1}")

            except Exception as e:
                logger.warning(f"  [{index:02d}/{total}] ✗ {fname} — {e}, retry {attempt+1}/{self.MAX_RETRIES}")

            await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))

        logger.error(f"  [{index:02d}/{total}] FAIL: {fname}")
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

                        total = len(files)
                        tasks = [
                            self.upload_file(session, fp, product_id, category, i+1, total)
                            for i, fp in enumerate(files)
                        ]
                        results = await asyncio.gather(*tasks)
                        success = sum(results)
                        failed  = total - success

                        if failed == 0:
                            logger.success(f"  Hoàn thành {product_id}: {success}/{total} ảnh")
                            os.remove(job_file)
                            await self._notify_server(session, product_id,
                                                      category, job['metadata'])
                        else:
                            logger.warning(f"  {failed}/{total} ảnh thất bại — giữ job để retry sau")

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


    def scan_and_enqueue(self, save_dir: str, category: str = "san_pham"):
        """Quét thư mục ảnh đã chụp, tự enqueue tất cả sản phẩm chưa upload."""
        save_path = Path(save_dir)
        if not save_path.exists():
            return 0

        # Lấy danh sách product_id đang có trong queue (tránh enqueue lại)
        queued = set()
        for jf in self.queue_dir.glob("*.json"):
            try:
                job = json.loads(jf.read_text())
                queued.add(job.get("product_id", ""))
            except Exception:
                pass

        count = 0
        for product_dir in sorted(save_path.iterdir()):
            if not product_dir.is_dir():
                continue
            product_id = product_dir.name
            if product_id in queued:
                logger.info(f"  [Skip] {product_id} — đã có trong queue")
                continue
            files = sorted(str(f) for f in product_dir.glob("*.jpg"))
            if not files:
                continue
            self.enqueue(files, product_id, category=category)
            count += 1

        return count


if __name__ == "__main__":
    engine   = TransferEngine()
    save_dir = os.getenv("SAVE_DIR", "/home/haiprotn/Documents/detect_product_ai/data/products")
    category = os.getenv("CATEGORY", "san_pham")

    logger.info(f"[Transfer] Server: {TransferEngine.SERVER_URL}")
    logger.info(f"[Transfer] Queue:  {engine.queue_dir}")

    # Tự quét ảnh đã chụp khi khởi động
    found = engine.scan_and_enqueue(save_dir, category)
    if found:
        logger.info(f"[Transfer] Tìm thấy {found} sản phẩm chưa upload → đã thêm vào queue")
    else:
        logger.info(f"[Transfer] Không có sản phẩm mới trong {save_dir}")

    logger.info("[Transfer] Bắt đầu upload... (Ctrl+C để dừng)\n")
    try:
        asyncio.run(engine.process_queue())
    except KeyboardInterrupt:
        logger.info("[Transfer] Dừng.")
