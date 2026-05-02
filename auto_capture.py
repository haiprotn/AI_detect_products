# auto_capture.py — Logic tự động chụp đủ góc
import os
os.environ.setdefault("DISPLAY", ":0")
import cv2
import time
import uuid
import hashlib
import numpy as np
from pathlib import Path
from loguru import logger
from quality_gate import JetsonQualityGate, QualityReport

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
        read_failures = 0

        logger.info(f"[Session] Bắt đầu: {product_id}")

        if show_ui:
            cv2.namedWindow('Jetson Capture Station', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Jetson Capture Station', 1280, 720)

        while len(saved_files) < self.TARGET_PER_SESSION:
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures >= self.MAX_READ_FAILURES:
                    logger.error("Camera mất kết nối, dừng session")
                    break
                continue
            read_failures = 0

            frame = cv2.resize(frame, (self.PROCESS_WIDTH, self.PROCESS_HEIGHT),
                               interpolation=cv2.INTER_AREA)

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
                save_frame = self.gate.remove_background(
                    frame, report.mask, report.object_bbox
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

        # Thử theo thứ tự ưu tiên: HW decode Jetson → SW decode → FFmpeg
        pipelines = [
            # Pipeline 1: NVIDIA HW decode — độ trễ thấp nhất trên Jetson
            (
                f"rtspsrc location={safe_url} latency=0 buffer-mode=0 ! "
                "rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true sync=false",
                cv2.CAP_GSTREAMER
            ),
            # Pipeline 2: GStreamer SW decode
            (
                f"rtspsrc location={safe_url} latency=0 ! "
                "decodebin ! videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink max-buffers=1 drop=true sync=false",
                cv2.CAP_GSTREAMER
            ),
            # Pipeline 3: FFmpeg (CAP_FFMPEG) — ổn định nhất khi GStreamer lỗi
            (safe_url, cv2.CAP_FFMPEG),
            # Pipeline 4: để OpenCV tự chọn
            (safe_url, cv2.CAP_ANY),
        ]

        for source, backend in pipelines:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                # Giảm buffer để lấy frame mới nhất, giảm độ trễ
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
        
        # Vẽ bbox nếu có
        if report.object_bbox:
            cx, cy, bw, bh = report.object_bbox
            x1, y1 = int(cx-bw/2), int(cy-bh/2)
            cv2.rectangle(display, (x1,y1), (x1+bw, y1+bh), color, 2)
        
        return display