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
    object_bbox:   Optional[Tuple]   # (x1, y1, x2, y2) vùng crop cố định
    mask:          Optional[np.ndarray]
    reject_reason: str
    motion_score:  float = 0.0


class JetsonQualityGate:
    """
    Không dùng YOLO detect — crop vùng trung tâm cố định.
    Người dùng đặt sản phẩm vào khung vuông hiển thị trên màn hình.
    Chỉ kiểm tra blur + brightness trên vùng đó.
    """

    BLUR_MIN       = 80.0   # ngưỡng độ nét trên vùng crop
    BRIGHTNESS_MIN = 40
    BRIGHTNESS_MAX = 230
    MOTION_MAX     = 20.0
    CROP_RATIO     = 0.55   # khung vuông chiếm 55% cạnh nhỏ của frame

    def __init__(self):
        self._prev_gray = None
        print("[QualityGate] Ready — center crop mode (không cần YOLO)")

    def _get_crop_box(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Tính vùng crop vuông cố định ở giữa frame."""
        side = int(min(h, w) * self.CROP_RATIO)
        cx, cy = w // 2, h // 2
        x1 = cx - side // 2
        y1 = cy - side // 2
        x2 = cx + side // 2
        y2 = cy + side // 2
        return x1, y1, x2, y2

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
                    reject_reason=f"Đang di chuyển ({motion_score:.1f}) — giữ yên"
                )
        self._prev_gray = gray

        # ── Lấy vùng crop trung tâm ───────────────────────
        x1, y1, x2, y2 = self._get_crop_box(h, w)
        roi  = gray[y1:y2, x1:x2]
        bbox = (x1, y1, x2, y2)

        # ── Blur check trên vùng crop ─────────────────────
        blur_score = cv2.Laplacian(roi, cv2.CV_64F).var()
        if blur_score < self.BLUR_MIN:
            return QualityReport(
                passed=False, blur_score=blur_score, brightness=0,
                has_object=True, object_bbox=bbox, mask=None,
                reject_reason=f"Mờ ({blur_score:.0f}) — giữ yên tay"
            )

        # ── Brightness check ──────────────────────────────
        brightness = roi.mean()
        if brightness < self.BRIGHTNESS_MIN:
            return QualityReport(
                passed=False, blur_score=blur_score, brightness=brightness,
                has_object=True, object_bbox=bbox, mask=None,
                reject_reason=f"Quá tối ({brightness:.0f})"
            )
        if brightness > self.BRIGHTNESS_MAX:
            return QualityReport(
                passed=False, blur_score=blur_score, brightness=brightness,
                has_object=True, object_bbox=bbox, mask=None,
                reject_reason=f"Quá sáng ({brightness:.0f})"
            )

        return QualityReport(
            passed=True, blur_score=blur_score, brightness=brightness,
            has_object=True, object_bbox=bbox, mask=None,
            reject_reason=""
        )

    def crop_object(self, frame: np.ndarray,
                    bbox: Optional[Tuple],
                    padding: float = 0.0,
                    square: bool = True) -> np.ndarray:
        """Crop vùng bbox và resize về 224x224."""
        if bbox is None:
            h, w = frame.shape[:2]
            bbox = self._get_crop_box(h, w)
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
