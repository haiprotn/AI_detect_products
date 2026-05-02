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
    MIN_INTERVAL_SEC    = 0.2  # 50 ảnh / 10 giây
    MAX_READ_FAILURES   = 30   # Số frame lỗi tối đa trước khi dừng
    
    PHASES = [
        {"name": "Mặt trước",  "target": 10, "instruction": "Để sản phẩm nhìn thẳng vào camera"},
        {"name": "Góc 45°",    "target": 8,  "instruction": "Xoay sản phẩm 45 độ"},
        {"name": "Góc 90°",    "target": 8,  "instruction": "Xoay thêm 45 độ nữa"},
        {"name": "Góc 135°",   "target": 8,  "instruction": "Tiếp tục xoay"},
        {"name": "Mặt sau",    "target": 8,  "instruction": "Lật mặt sau lên"},
        {"name": "Mặt trên",   "target": 8,  "instruction": "Nghiêng để thấy mặt trên"},
    ]
    
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
        phase_idx    = 0
        phase_count  = 0

        # Dùng grabber thread để luôn có frame mới nhất, tránh buffer tích lũy
        grabber = RTSPFrameGrabber(cap)
        grabber.start()

        logger.info(f"[Session] Bắt đầu: {product_id}")

        # Chờ frame đầu tiên
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

            # Che vùng OSD camera (timestamp góc trên, logo góc dưới)
            h_f, w_f = frame.shape[:2]
            frame[:int(h_f * 0.08), :int(w_f * 0.35)] = 128   # góc trên trái
            frame[int(h_f * 0.92):, int(w_f * 0.65):]  = 128   # góc dưới phải

            current_phase = self.PHASES[min(phase_idx, len(self.PHASES)-1)]

            # Kiểm tra chất lượng
            report = self.gate.check(frame)
            if show_ui:
                display = self._draw_ui(
                    frame, report, current_phase,
                    len(saved_files), phase_count,
                    current_phase["target"]
                )
                cv2.imshow('Jetson Capture Station', display)

            # Logic lưu ảnh
            now = time.time()
            interval_ok = (now - self._last_capture_time) > self.MIN_INTERVAL_SEC

            if report.passed and interval_ok and not self._is_duplicate(frame):
                # Xóa nền + crop sản phẩm
                save_frame = self.gate.crop_object(
                    frame, report.object_bbox, padding=0.15
                )

                fname = f"{product_id}_{phase_idx:02d}_{phase_count:03d}.jpg"
                fpath = product_dir / fname
                cv2.imwrite(str(fpath), save_frame,
                           [cv2.IMWRITE_JPEG_QUALITY, 92])

                saved_files.append(str(fpath))
                phase_count             += 1
                self._last_capture_time  = now
                self._update_hash(frame)  # cập nhật hash sau khi chụp
                
                logger.debug(f"  Saved [{len(saved_files)}/{self.TARGET_PER_SESSION}]: {fname}")
                
                # Chuyển phase khi đủ ảnh
                if phase_count >= current_phase["target"]:
                    phase_idx  += 1
                    phase_count = 0
                    logger.info(f"  → Phase mới: {self.PHASES[min(phase_idx, len(self.PHASES)-1)]['name']}")
            
            if show_ui:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.warning("Session bị hủy bởi người dùng")
                    break
                if key == ord('s'):  # Skip phase thủ công
                    if phase_idx < len(self.PHASES) - 1:
                        phase_idx  += 1
                        phase_count = 0
                        logger.info(f"  → Skip tới: {self.PHASES[phase_idx]['name']}")
                    else:
                        logger.warning("Đã ở phase cuối, không thể skip thêm")
        
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

    def _draw_ui(self, frame, report, phase,
                 total_saved, phase_saved, phase_target):
        """Vẽ UI hướng dẫn lên màn hình"""
        display = cv2.resize(frame, (self.DISPLAY_W, self.DISPLAY_H))
        h, w = display.shape[:2]
        
        # Thanh tiến trình tổng
        progress = total_saved / self.TARGET_PER_SESSION
        bar_w = int(w * 0.8)
        cv2.rectangle(display, (int(w*0.1), h-60), 
                      (int(w*0.1) + bar_w, h-30), (50,50,50), -1)
        cv2.rectangle(display, (int(w*0.1), h-60),
                      (int(w*0.1) + int(bar_w * progress), h-30), 
                      (0,200,0), -1)
        
        # Thông tin phase
        cv2.rectangle(display, (0,0), (w, 80), (0,0,0), -1)
        cv2.putText(display, f"Phase: {phase['name']}  ({phase_saved}/{phase_target})",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        cv2.putText(display, phase['instruction'],
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        
        # Trạng thái quality
        if report.passed:
            color = (0, 255, 0)
            status = f"OK | Blur:{report.blur_score:.0f} | Bright:{report.brightness:.0f}"
        else:
            color = (0, 0, 255)
            status = f"REJECT: {report.reject_reason}"
        
        cv2.putText(display, status, (10, h-70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Tong: {total_saved}/{self.TARGET_PER_SESSION}",
                   (w-250, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # Khung vuông hướng dẫn đặt sản phẩm (cố định ở giữa màn hình)
        sq    = int(min(w, h) * 0.55)   # cạnh khung vuông = 55% cạnh nhỏ hơn
        cx_s  = w // 2
        cy_s  = h // 2
        x1_s  = cx_s - sq // 2
        y1_s  = cy_s - sq // 2
        x2_s  = cx_s + sq // 2
        y2_s  = cy_s + sq // 2

        guide_color = (0, 255, 0) if report.passed else (0, 100, 255)
        cv2.rectangle(display, (x1_s, y1_s), (x2_s, y2_s), guide_color, 2)
        # Vẽ 4 góc nổi bật hơn
        corner = sq // 8
        for px, py in [(x1_s, y1_s), (x2_s, y1_s),
                       (x1_s, y2_s), (x2_s, y2_s)]:
            dx = corner if px == x1_s else -corner
            dy = corner if py == y1_s else -corner
            cv2.line(display, (px, py), (px + dx, py), guide_color, 4)
            cv2.line(display, (px, py), (px, py + dy), guide_color, 4)

        # Bbox YOLO detect được
        if report.object_bbox:
            cx, cy, bw, bh = report.object_bbox
            side = max(bw, bh)
            half = side // 2
            cv2.rectangle(display,
                          (int(cx)-half, int(cy)-half),
                          (int(cx)+half, int(cy)+half),
                          (255, 200, 0), 1)

        return display