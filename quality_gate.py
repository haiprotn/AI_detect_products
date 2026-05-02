# quality_gate.py — Chạy trên Jetson
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class QualityReport:
    passed:        bool
    blur_score:    float
    brightness:    float
    has_object:    bool
    object_bbox:   Optional[Tuple]   # (cx, cy, w, h)
    mask:          Optional[np.ndarray]  # segmentation mask (H, W) uint8
    reject_reason: str
    motion_score:  float = 0.0


class JetsonQualityGate:
    """
    Chạy trên GPU Jetson — kiểm tra + xóa nền ảnh TRƯỚC khi lưu.
    Dùng YOLOv8n-seg để vừa detect vừa lấy mask sản phẩm.
    """

    BLUR_MIN       = 60.0
    BRIGHTNESS_MIN = 50
    BRIGHTNESS_MAX = 220
    OBJECT_CONF    = 0.40
    MOTION_MAX     = 25.0
    BG_COLOR       = 255   # màu nền sau khi xóa (255=trắng, 0=đen)

    # Class COCO cần loại bỏ — không phải sản phẩm
    EXCLUDE_CLASSES = {0}  # 0=person (tay, người cầm sản phẩm)

    def __init__(self):
        self._prev_gray = None
        self._load_detector()
        print("[QualityGate] Ready on Jetson GPU")

    def _load_detector(self):
        try:
            from ultralytics import YOLO
            self.detector     = YOLO("yolov8n.pt")   # detect only, nhẹ hơn seg
            self.use_seg      = False
            self.use_detector = True
            print("[QualityGate] YOLOv8n loaded")
        except Exception as e:
            print(f"[QualityGate] WARN: {e} — bỏ qua detect")
            self.use_detector = False
            self.use_seg      = False

    def check(self, frame: np.ndarray,
              check_motion: bool = False) -> QualityReport:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Motion ────────────────────────────────────────
        motion_score = 0.0
        if self._prev_gray is not None:
            motion_score = float(cv2.absdiff(gray, self._prev_gray).mean())
            if check_motion and motion_score > self.MOTION_MAX:
                self._prev_gray = gray
                return QualityReport(
                    passed=False, blur_score=0, brightness=0,
                    has_object=False, object_bbox=None, mask=None,
                    motion_score=motion_score,
                    reject_reason=f"Có chuyển động ({motion_score:.1f})"
                )
        self._prev_gray = gray

        # ── Blur ──────────────────────────────────────────
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.BLUR_MIN:
            return QualityReport(
                passed=False, blur_score=blur_score, brightness=0,
                has_object=False, object_bbox=None, mask=None,
                reject_reason=f"Ảnh mờ ({blur_score:.0f} < {self.BLUR_MIN})"
            )

        # ── Brightness ────────────────────────────────────
        brightness = gray.mean()
        if not (self.BRIGHTNESS_MIN <= brightness <= self.BRIGHTNESS_MAX):
            return QualityReport(
                passed=False, blur_score=blur_score, brightness=brightness,
                has_object=False, object_bbox=None, mask=None,
                reject_reason=f"Độ sáng không phù hợp ({brightness:.0f})"
            )

        # ── Detect + Seg ──────────────────────────────────
        bbox, mask = None, None
        has_object = True

        if self.use_detector:
            results = self.detector(frame, verbose=False,
                                    conf=self.OBJECT_CONF)
            boxes = results[0].boxes

            # Lọc bỏ class không phải sản phẩm (tay, người...)
            valid_indices = [
                i for i, cls in enumerate(boxes.cls.cpu().numpy().astype(int))
                if cls not in self.EXCLUDE_CLASSES
            ]

            if not valid_indices:
                return QualityReport(
                    passed=False, blur_score=blur_score,
                    brightness=brightness, has_object=False,
                    object_bbox=None, mask=None,
                    reject_reason="Không thấy sản phẩm (chỉ thấy tay/người)"
                )

            # Lấy object lớn nhất trong các object hợp lệ
            xywh       = boxes.xywh.cpu().numpy()
            valid_xywh = xywh[valid_indices]
            best_local = int(np.argmax(valid_xywh[:, 2] * valid_xywh[:, 3]))
            best_i     = valid_indices[best_local]
            largest    = xywh[best_i]
            bbox       = tuple(largest.astype(int))

            # Lấy segmentation mask tương ứng
            if self.use_seg and results[0].masks is not None:
                raw_mask = results[0].masks.data[best_i].cpu().numpy()
                mask = cv2.resize(raw_mask, (w, h),
                                  interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8)

            # Center check
            cx, cy = bbox[0], bbox[1]
            if (abs(cx - w/2) / (w/2) > 0.40 or
                    abs(cy - h/2) / (h/2) > 0.40):
                return QualityReport(
                    passed=False, blur_score=blur_score,
                    brightness=brightness, has_object=True,
                    object_bbox=bbox, mask=None,
                    reject_reason="Sản phẩm lệch khỏi trung tâm"
                )

        return QualityReport(
            passed=True, blur_score=blur_score, brightness=brightness,
            has_object=True, object_bbox=bbox, mask=mask,
            reject_reason=""
        )

    def crop_object(self, frame: np.ndarray,
                    bbox: Optional[Tuple],
                    padding: float = 0.15,
                    square: bool = True) -> np.ndarray:
        """Crop vùng sản phẩm theo bbox YOLO + padding. Output hình vuông nếu square=True."""
        if bbox is None:
            return frame
        h, w = frame.shape[:2]
        cx, cy, bw, bh = bbox

        if square:
            # Lấy cạnh lớn nhất để tạo hình vuông
            side  = max(bw, bh)
            pad   = int(side * padding)
            half  = side // 2 + pad
            x1 = max(0, int(cx) - half)
            y1 = max(0, int(cy) - half)
            x2 = min(w, int(cx) + half)
            y2 = min(h, int(cy) + half)
        else:
            pad_x = int(bw * padding)
            pad_y = int(bh * padding)
            x1 = max(0, int(cx - bw/2) - pad_x)
            y1 = max(0, int(cy - bh/2) - pad_y)
            x2 = min(w, int(cx + bw/2) + pad_x)
            y2 = min(h, int(cy + bh/2) + pad_y)

        cropped = frame[y1:y2, x1:x2]
        # Resize về 224x224 chuẩn cho DINOv2
        return cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
