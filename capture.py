"""
capture.py — Jetson Edge
Pipeline: Camera → YOLOv8 detect → crop → blur check → lưu ảnh → upload
"""
import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

SAVE_DIR   = Path(os.getenv("SAVE_DIR", "/home/user/data/products"))
CAM_SOURCE = os.getenv("RTSP_URL") or int(os.getenv("CAM_INDEX", 0))
CATEGORY   = os.getenv("CATEGORY", "san_pham")
TARGET     = 50
BLUR_MIN   = 80.0
INTERVAL   = 0.5
CROP_RATIO = 0.60    # fallback nếu YOLO không detect được
IMG_SIZE   = 224
DISPLAY_W  = 960
DISPLAY_H  = 540

# Lọc object quá lớn (nền, thân người) — chiếm > 45% diện tích frame
MAX_OBJ_RATIO = 0.45
# Lọc class COCO: 0=person, 67=cell phone ... (giữ lại hầu hết vật thể nhỏ)
EXCLUDE_CLS   = set()   # để trống = không lọc theo class, chỉ lọc theo size


# ── YOLO ─────────────────────────────────────────────────────────────────────

_yolo = None

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO("yolov8n.pt")
        logger.info("YOLOv8n loaded")
    return _yolo


def detect_product(frame: np.ndarray):
    """
    Trả về (x1,y1,x2,y2) bbox sản phẩm hoặc None.
    Ưu tiên object nhỏ nhất (sản phẩm), loại object quá lớn (nền/người).
    """
    h, w = frame.shape[:2]
    frame_area = h * w
    try:
        results = get_yolo()(frame, verbose=False, conf=0.25)
        boxes   = results[0].boxes
        if len(boxes) == 0:
            return None

        valid = []
        for i, box in enumerate(boxes.xywh.cpu().numpy()):
            cx, cy, bw, bh = box
            cls = int(boxes.cls[i].item())
            area_ratio = (bw * bh) / frame_area
            if area_ratio > MAX_OBJ_RATIO:
                continue
            if cls in EXCLUDE_CLS:
                continue
            valid.append((area_ratio, cx, cy, bw, bh))

        if not valid:
            return None

        # Lấy object nhỏ nhất (sản phẩm, không phải nền)
        valid.sort(key=lambda x: x[0])
        _, cx, cy, bw, bh = valid[0]

        pad = int(max(bw, bh) * 0.15)
        x1  = max(0, int(cx - bw/2) - pad)
        y1  = max(0, int(cy - bh/2) - pad)
        x2  = min(w, int(cx + bw/2) + pad)
        y2  = min(h, int(cy + bh/2) + pad)
        return x1, y1, x2, y2

    except Exception as e:
        logger.warning(f"YOLO error: {e}")
        return None


def fallback_crop(h: int, w: int):
    """Crop trung tâm khi YOLO không detect được."""
    side = int(min(h, w) * CROP_RATIO)
    cx, cy = w//2, h//2
    return cx-side//2, cy-side//2, cx+side//2, cy+side//2


# ── Frame Grabber ─────────────────────────────────────────────────────────────

class FrameGrabber(threading.Thread):
    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.frame  = None
        self._lock  = threading.Lock()
        self._stop  = False

    def run(self):
        cap = self._open()
        while not self._stop:
            try:
                ok, f = cap.read()
                if ok:
                    with self._lock:
                        self.frame = f
                else:
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.1)
        cap.release()

    def _open(self):
        if isinstance(self.source, str):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|threads;1"
            )
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def latest(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stop = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def blur_score(roi_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())


def phash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (8, 8)).astype(float)
    return small.flatten() > small.mean()


def draw(frame, bbox, passed, blur, saved, reason=""):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    blur_bg = cv2.GaussianBlur(frame, (31, 31), 0)
    mask    = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = 255
    out = np.where(mask[:,:,None] == 255, frame, blur_bg)

    color = (0, 220, 0) if passed else (0, 80, 255)
    cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
    c = (x2-x1)//8
    for px, py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        dx = c if px==x1 else -c
        dy = c if py==y1 else -c
        cv2.line(out, (px,py), (px+dx,py), color, 4)
        cv2.line(out, (px,py), (px,py+dy), color, 4)

    tag = "YOLO" if bbox != fallback_crop(h, w) else "CENTER"
    cv2.putText(out, tag, (x1+4, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

    cv2.rectangle(out, (0,0), (w,52), (0,0,0), -1)
    status = f"OK  blur={blur:.0f}" if passed else (reason or "Chua san sang")
    cv2.putText(out, status, (12,34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(out, f"{saved}/{TARGET}", (w-160,34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    bx, bw_ = int(w*0.1), int(w*0.8)
    cv2.rectangle(out, (bx,h-35), (bx+bw_,h-12), (50,50,50), -1)
    cv2.rectangle(out, (bx,h-35), (bx+int(bw_*saved/TARGET),h-12), (0,200,0), -1)

    return cv2.resize(out, (DISPLAY_W, DISPLAY_H))


# ── Session ───────────────────────────────────────────────────────────────────

def run_session(product_id: str) -> list:
    product_dir = SAVE_DIR / CATEGORY / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    grabber = FrameGrabber(CAM_SOURCE)
    grabber.start()
    logger.info("Kết nối camera...")
    while grabber.latest() is None:
        time.sleep(0.05)

    cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture", DISPLAY_W, DISPLAY_H)

    saved     = []
    last_hash = None
    last_time = 0.0
    logger.info(f"[{product_id}] Bắt đầu — đưa sản phẩm vào khung")

    while len(saved) < TARGET:
        frame = grabber.latest()
        if frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]

        # YOLO detect → crop, fallback center crop nếu không detect
        det  = detect_product(frame)
        bbox = det if det else fallback_crop(h, w)
        x1, y1, x2, y2 = bbox
        roi  = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        b    = blur_score(gray)
        br   = float(gray.mean())

        passed, reason = True, ""
        if det is None:
            passed, reason = False, "Khong thay san pham — dua vao khung"
        elif b < BLUR_MIN:
            passed, reason = False, f"Mo ({b:.0f}) — giu yen"
        elif br < 30:
            passed, reason = False, "Qua toi"
        elif br > 240:
            passed, reason = False, "Qua sang"

        cv2.imshow("Capture", draw(frame, bbox, passed, b, len(saved), reason))

        now = time.time()
        if passed and (now - last_time) >= INTERVAL:
            cur_hash = phash(gray)
            is_dup   = (last_hash is not None and
                        np.mean(cur_hash == last_hash) > 0.97)
            if not is_dup:
                # Lấy frame nét nhất trong 0.3s
                best, best_b = roi.copy(), b
                t_end = time.time() + 0.3
                while time.time() < t_end:
                    f2 = grabber.latest()
                    if f2 is not None:
                        r2 = f2[y1:y2, x1:x2]
                        g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)
                        s2 = blur_score(g2)
                        if s2 > best_b:
                            best, best_b = r2.copy(), s2
                    time.sleep(0.02)

                save = cv2.resize(best, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_AREA)
                fname = f"{product_id}_{len(saved):03d}.jpg"
                cv2.imwrite(str(product_dir/fname), save,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved.append(str(product_dir/fname))
                last_hash = cur_hash
                last_time = now
                logger.info(f"  [{len(saved):02d}/{TARGET}] {fname}  blur={best_b:.0f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.warning("Hủy session")
            break

    grabber.stop()
    cv2.destroyAllWindows()
    logger.info(f"Hoàn thành: {len(saved)} ảnh → {product_dir}")
    return saved


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 50)
    print("  Jetson Capture — YOLOv8 + DINOv2-base")
    print(f"  Camera : {CAM_SOURCE}")
    print(f"  Lưu vào: {SAVE_DIR / CATEGORY}")
    print("  Q = thoát session")
    print("=" * 50)

    while True:
        try:
            pid = input("\nNhập tên sản phẩm (q=thoát): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pid.lower() == "q" or not pid:
            break
        files = run_session(pid)
        if files:
            from uploader import enqueue
            enqueue(files, pid, CATEGORY)
            print(f"→ Đã thêm vào hàng đợi upload ({len(files)} ảnh)")
            print("  Chạy: python uploader.py để upload lên server")
