# auto_capture.py — Logic tự động chụp đủ góc
import os
os.environ.setdefault("DISPLAY", ":0")
import cv2
import time
import threading
import hashlib
import numpy as np
from pathlib import Path
from loguru import logger
from quality_gate import JetsonQualityGate, QualityReport


class RTSPFrameGrabber(threading.Thread):
    """Thread riêng grab frame liên tục — chỉ giữ frame mới nhất, xóa buffer cũ."""
    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self._cap   = cap
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = False

    def run(self):
        while not self._stop:
            try:
                ok, frame = self._cap.read()
                if ok:
                    with self._lock:
                        self._frame = frame
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.05)

    def latest(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._stop = True

class AutoCaptureController:
    """
    Tự động chụp 50 ảnh/sản phẩm theo session
    Hướng dẫn nhân viên xoay sản phẩm qua màn hình
    """
    
    TARGET_PER_SESSION  = 50   # Số ảnh mỗi sản phẩm
    MIN_INTERVAL_SEC    = 0.3  # giây tối thiểu giữa 2 lần chụp
    
    PROCESS_WIDTH  = 1280  # Resize về độ rộng này trước khi xử lý
    PROCESS_HEIGHT = 720

    def __init__(self, quality_gate: JetsonQualityGate, save_dir: str):
        self.gate       = quality_gate
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._last_frame_hash = None    # hash của frame đã chụp gần nhất
        self._last_capture_time = 0
    
    def _frame_hash(self, frame: np.ndarray) -> np.ndarray:
        """pHash 8x8. Ảnh uniform (std≈0) dùng brightness level thay thế."""
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (8, 8)).astype(float)
        mean = small.mean()
        if small.std() < 1e-6:
            # Encode brightness vào bit pattern để phân biệt ảnh uniform khác nhau
            level = int(mean) >> 2  # 0-63
            bits = np.zeros(64, dtype=bool)
            bits[:level] = True
            return bits
        return (small.flatten() > mean)

    STATIC_THRESH = 0.99  # Chỉ reject khi gần như không đổi gì

    def _is_duplicate(self, frame: np.ndarray) -> bool:
        """Chỉ reject khi sản phẩm đứng YÊN HOÀN TOÀN so với ảnh vừa chụp."""
        current_hash = self._frame_hash(frame)
        if self._last_frame_hash is not None:
            similarity = np.mean(current_hash == self._last_frame_hash)
            if similarity >= self.STATIC_THRESH:
                return True
        return False

    def _update_hash(self, frame: np.ndarray):
        """Gọi sau khi chụp xong để cập nhật hash tham chiếu."""
        self._last_frame_hash = self._frame_hash(frame)
    
    def run_session(self, product_id: str,
                    cam_source: int | str = 0,
                    show_ui: bool = False) -> list:
        """
        Chạy 1 session thu thập cho 1 sản phẩm
        Trả về list đường dẫn ảnh đã lưu
        """
        product_dir = self.save_dir / product_id
        product_dir.mkdir(exist_ok=True)
        
        cap = self._open_camera(cam_source)
        saved_files = []

        grabber = RTSPFrameGrabber(cap)
        grabber.start()

        logger.info(f"[Session] Bắt đầu: {product_id}")

        while grabber.latest() is None:
            time.sleep(0.05)

        if show_ui:
            cv2.namedWindow('Jetson Capture Station', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Jetson Capture Station', 1280, 720)

        while len(saved_files) < self.TARGET_PER_SESSION:
            frame = grabber.latest()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (self.PROCESS_WIDTH, self.PROCESS_HEIGHT),
                               interpolation=cv2.INTER_AREA)

            # Che OSD camera
            h_f, w_f = frame.shape[:2]
            frame[:int(h_f * 0.08), :int(w_f * 0.35)] = 128
            frame[int(h_f * 0.92):, int(w_f * 0.65):]  = 128

            report = self.gate.check(frame)

            if show_ui:
                display = self._draw_ui(frame, report, len(saved_files))
                cv2.imshow('Jetson Capture Station', display)

            now         = time.time()
            interval_ok = (now - self._last_capture_time) > self.MIN_INTERVAL_SEC

            if report.passed and interval_ok and not self._is_duplicate(frame):
                # Chờ 0.4s lấy frame nét nhất
                best_frame = frame
                best_blur  = report.blur_score
                deadline   = time.time() + 0.4
                while time.time() < deadline:
                    cand = grabber.latest()
                    if cand is not None:
                        gray  = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
                        score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        if score > best_blur:
                            best_blur  = score
                            best_frame = cand
                    time.sleep(0.02)

                save_frame = self.gate.crop_object(
                    best_frame, report.object_bbox, padding=0.15
                )

                fname = f"{product_id}_{len(saved_files):03d}.jpg"
                fpath = product_dir / fname
                cv2.imwrite(str(fpath), save_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

                saved_files.append(str(fpath))
                self._last_capture_time = now
                self._update_hash(frame)
                logger.info(f"  [{len(saved_files):02d}/{self.TARGET_PER_SESSION}] {fname} blur={best_blur:.0f}")

            if show_ui:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.warning("Session bị hủy")
                    break
        
        grabber.stop()
        cap.release()
        if show_ui:
            cv2.destroyAllWindows()
        logger.info(f"[Session] Hoàn thành: {len(saved_files)} ảnh")
        return saved_files
    
    def _open_camera(self, source: int | str):
        if isinstance(source, str) and source.startswith("rtsp://"):
            return self._open_rtsp(source)

        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở camera: {source}")
        return cap

    @staticmethod
    def _encode_rtsp_url(url: str) -> str:
        """Encode @ trong password để tránh parse sai khi password có ký tự @."""
        # rtsp://user:pass@host:port/path
        prefix = "rtsp://"
        rest = url[len(prefix):]          # user:pass@host:port/path
        # Tìm vị trí @ cuối cùng — đó là @ phân cách credentials và host
        last_at = rest.rfind("@")
        if last_at == -1:
            return url
        credentials = rest[:last_at]      # user:pass (có thể chứa @)
        hostpath    = rest[last_at + 1:]  # host:port/path
        # Encode @ trong password thành %40
        if ":" in credentials:
            user, password = credentials.split(":", 1)
            credentials = f"{user}:{password.replace('@', '%40')}"
        return f"{prefix}{credentials}@{hostpath}"

    def _open_rtsp(self, url: str):
        safe_url = self._encode_rtsp_url(url)

        # Force TCP + tắt async decode để tránh xung đột với Python thread
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "fflags;nobuffer|"
            "flags;low_delay|"
            "threads;1"
        )

        pipelines = [
            # Pipeline 1: NVIDIA HW decode Jetson
            (
                f"rtspsrc location={safe_url} latency=0 protocols=tcp ! "
                "rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true sync=false",
                cv2.CAP_GSTREAMER
            ),
            # Pipeline 2: GStreamer SW decode
            (
                f"rtspsrc location={safe_url} latency=0 protocols=tcp ! "
                "decodebin ! videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true sync=false",
                cv2.CAP_GSTREAMER
            ),
            # Pipeline 3: FFmpeg single-thread
            (safe_url, cv2.CAP_FFMPEG),
        ]

        for source, backend in pipelines:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info(f"[Camera] RTSP kết nối thành công: {url}")
                return cap
            cap.release()

        raise RuntimeError(f"Không thể kết nối RTSP: {url}")
    
    DISPLAY_W = 1280
    DISPLAY_H = 700  # để chừa taskbar trên màn hình 1366x768

    def _draw_ui(self, frame, report, total_saved):
        """Vẽ UI đơn giản — không có phase."""
        display = cv2.resize(frame, (self.DISPLAY_W, self.DISPLAY_H))
        h, w    = display.shape[:2]

        # Header
        cv2.rectangle(display, (0, 0), (w, 55), (0, 0, 0), -1)
        status_txt = (f"OK | Blur:{report.blur_score:.0f}" if report.passed
                      else f"REJECT: {report.reject_reason}")
        color = (0, 255, 0) if report.passed else (0, 80, 255)
        cv2.putText(display, status_txt, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(display, f"{total_saved}/{self.TARGET_PER_SESSION}",
                    (w - 200, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

        # Thanh tiến trình
        bar_x  = int(w * 0.1)
        bar_w  = int(w * 0.8)
        prog_w = int(bar_w * total_saved / self.TARGET_PER_SESSION)
        cv2.rectangle(display, (bar_x, h - 40), (bar_x + bar_w, h - 15), (50, 50, 50), -1)
        cv2.rectangle(display, (bar_x, h - 40), (bar_x + prog_w, h - 15), (0, 200, 0), -1)
        
        # Khung vuông cố định — đúng vùng sẽ được crop và lưu
        sq     = int(min(w, h) * 0.55)
        cx_s   = w // 2
        cy_s   = h // 2
        x1_s   = cx_s - sq // 2
        y1_s   = cy_s - sq // 2
        x2_s   = cx_s + sq // 2
        y2_s   = cy_s + sq // 2
        gc     = (0, 255, 0) if report.passed else (0, 100, 255)

        # Làm mờ vùng ngoài khung để tập trung nhìn vào sản phẩm
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1_s:y2_s, x1_s:x2_s] = 255
        blurred = cv2.GaussianBlur(display, (21, 21), 0)
        display = np.where(mask[:, :, None] == 255, display, blurred)

        cv2.rectangle(display, (x1_s, y1_s), (x2_s, y2_s), gc, 2)
        corner = sq // 8
        for px, py in [(x1_s, y1_s), (x2_s, y1_s),
                       (x1_s, y2_s), (x2_s, y2_s)]:
            dx = corner if px == x1_s else -corner
            dy = corner if py == y1_s else -corner
            cv2.line(display, (px, py), (px + dx, py), gc, 4)
            cv2.line(display, (px, py), (px, py + dy), gc, 4)

        cv2.putText(display, "Dat san pham vao khung",
                    (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, gc, 2)
        return display