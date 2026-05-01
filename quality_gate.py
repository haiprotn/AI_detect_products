# quality_gate.py — Chạy trên Jetson
import cv2
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class QualityReport:
    passed:         bool
    blur_score:     float   # Cao = nét
    brightness:     float   # 0-255
    has_object:     bool
    object_bbox:    Optional[Tuple]  # (x, y, w, h)
    reject_reason:  str
    motion_score:   float = 0.0  # 0 = tĩnh, cao = có chuyển động

class JetsonQualityGate:
    """
    Chạy trên GPU Jetson — kiểm tra ảnh TRƯỚC khi lưu
    Tiết kiệm bandwidth: chỉ gửi ảnh đạt chuẩn về server
    """
    
    BLUR_MIN       = 60.0    # Laplacian variance — hạ xuống cho phép lắc nhẹ
    BRIGHTNESS_MIN = 60
    BRIGHTNESS_MAX = 210
    OBJECT_CONF    = 0.5
    MOTION_MAX     = 25.0   # Pixel diff — 8=nhạy (chỉ cho bàn xoay), 25=xoay tay được

    def __init__(self):
        # Load YOLOv8n để detect vật thể
        # Model nhỏ nhất ~6MB, chạy ~30ms trên Jetson
        self._load_detector()
        self._prev_gray = None  # Frame trước để tính motion
        print("[QualityGate] Ready on Jetson GPU")
    
    def _load_detector(self):
        """YOLOv8n TensorRT — tối ưu cho Jetson"""
        try:
            from ultralytics import YOLO
            self.detector = YOLO('yolov8n.engine')  # TensorRT format
            self.use_detector = True
        except Exception:
            print("[WARN] Detector không load được, bỏ qua detect bước này")
            self.use_detector = False
    
    def check(self, frame: np.ndarray, check_motion: bool = False) -> QualityReport:
        """
        Kiểm tra chất lượng ảnh.
        check_motion=False (mặc định): bỏ qua motion — dùng khi đang xoay sản phẩm
        check_motion=True: reject khi có chuyển động đột ngột (phát hiện tay)
        """
        # ── 0. Motion Detection ───────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0
        if self._prev_gray is not None:
            motion_score = float(cv2.absdiff(gray, self._prev_gray).mean())
            if check_motion and motion_score > self.MOTION_MAX:
                self._prev_gray = gray
                return QualityReport(
                    passed=False, blur_score=0, brightness=0,
                    has_object=False, object_bbox=None,
                    motion_score=motion_score,
                    reject_reason=f"Có chuyển động ({motion_score:.1f}) — chờ tay rút ra"
                )
        self._prev_gray = gray

        # ── 1. Blur Detection ─────────────────────────
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < self.BLUR_MIN:
            return QualityReport(
                passed=False, blur_score=blur_score,
                brightness=0, has_object=False,
                object_bbox=None,
                reject_reason=f"Ảnh mờ ({blur_score:.0f} < {self.BLUR_MIN})"
            )
        
        # ── 2. Brightness Check ───────────────────────
        brightness = gray.mean()
        if brightness < self.BRIGHTNESS_MIN:
            return QualityReport(
                passed=False, blur_score=blur_score,
                brightness=brightness, has_object=False,
                object_bbox=None,
                reject_reason=f"Quá tối ({brightness:.0f})"
            )
        if brightness > self.BRIGHTNESS_MAX:
            return QualityReport(
                passed=False, blur_score=blur_score,
                brightness=brightness, has_object=False,
                object_bbox=None,
                reject_reason=f"Quá sáng ({brightness:.0f})"
            )
        
        # ── 3. Object Detection ───────────────────────
        bbox = None
        has_object = True  # True nếu bỏ qua detect; bị override → False nếu detect thất bại

        if self.use_detector:
            results = self.detector(frame, verbose=False, conf=self.OBJECT_CONF)
            if len(results[0].boxes) == 0:
                return QualityReport(
                    passed=False, blur_score=blur_score,
                    brightness=brightness, has_object=False,
                    object_bbox=None,
                    reject_reason="Không thấy vật thể trong khung"
                )
            
            # Lấy object lớn nhất (sản phẩm chính)
            boxes = results[0].boxes.xywh.cpu().numpy()
            largest = boxes[np.argmax(boxes[:, 2] * boxes[:, 3])]
            bbox = tuple(largest.astype(int))
        
        # ── 4. Center Check: object phải ở giữa khung ─
        if bbox:
            h, w = frame.shape[:2]
            cx, cy = bbox[0], bbox[1]
            center_x_ratio = abs(cx - w/2) / (w/2)
            center_y_ratio = abs(cy - h/2) / (h/2)
            
            if center_x_ratio > 0.35 or center_y_ratio > 0.35:
                return QualityReport(
                    passed=False, blur_score=blur_score,
                    brightness=brightness, has_object=True,
                    object_bbox=bbox,
                    reject_reason="Sản phẩm lệch khỏi trung tâm"
                )
        
        return QualityReport(
            passed=True, blur_score=blur_score,
            brightness=brightness, has_object=True,
            object_bbox=bbox, reject_reason=""
        )
    
    def crop_object(self, frame: np.ndarray,
                    bbox: Optional[Tuple], padding: float = 0.15) -> np.ndarray:
        """Cắt vùng sản phẩm + padding"""
        if bbox is None:
            return frame
        
        h, w = frame.shape[:2]
        cx, cy, bw, bh = bbox
        
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        
        x1 = max(0, int(cx - bw/2) - pad_x)
        y1 = max(0, int(cy - bh/2) - pad_y)
        x2 = min(w, int(cx + bw/2) + pad_x)
        y2 = min(h, int(cy + bh/2) + pad_y)
        
        return frame[y1:y2, x1:x2]