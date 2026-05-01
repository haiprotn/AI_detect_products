# test_auto_capture.py
import os
import sys
import contextlib
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from auto_capture import AutoCaptureController
from quality_gate import QualityReport

@contextlib.contextmanager
def suppress_c_stderr():
    """Ẩn stderr ở mức C (GStreamer, FFmpeg viết thẳng vào fd 2)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(devnull_fd)

import os
from dotenv import load_dotenv
load_dotenv()
RTSP_URL = os.getenv("RTSP_URL", "rtsp://user:pass@192.168.x.x:554/Streaming/Channels/101")

def make_gate():
    gate = MagicMock()
    gate.check.return_value = QualityReport(
        passed=True, blur_score=150, brightness=128,
        has_object=True, object_bbox=(320, 240, 100, 80), reject_reason=""
    )
    gate.crop_object.side_effect = lambda frame, bbox: frame
    return gate

def make_mock_cap(n_frames=200, fail_after=None):
    """Camera giả trả về frames rõ ràng khác nhau sau khi resize 8x8."""
    calls = [0]
    def read():
        i = calls[0]
        calls[0] += 1
        if fail_after and i >= fail_after:
            return False, None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Thay đổi toàn bộ nửa trên frame — đủ lớn để pHash phân biệt
        frame[:240, :] = (i * 5) % 256
        frame[240:, :] = 255 - (i * 5) % 256
        return True, frame
    cap = MagicMock()
    cap.read.side_effect = read
    return cap

# ── Unit tests ──────────────────────────────────────────────────

def test_frame_hash_detects_similar():
    ctrl = AutoCaptureController(make_gate(), "/tmp/test_capture")
    frame1 = np.full((480, 640, 3), 100, dtype=np.uint8)
    frame2 = frame1.copy()
    frame2[0, 0] = 101  # thay đổi 1 pixel

    ctrl._is_duplicate(frame1)
    assert ctrl._is_duplicate(frame2), "Ảnh gần giống phải bị detect là duplicate"
    print("✓ pHash phát hiện ảnh tương đồng")

def test_frame_hash_different():
    ctrl = AutoCaptureController(make_gate(), "/tmp/test_capture")
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.full((480, 640, 3), 200, dtype=np.uint8)

    ctrl._is_duplicate(frame1)
    assert not ctrl._is_duplicate(frame2), "Ảnh khác nhau không được coi là duplicate"
    print("✓ pHash không nhầm ảnh khác nhau")

def test_open_local_camera_raises_if_missing():
    """Mock VideoCapture để không thật sự thử mở camera."""
    ctrl = AutoCaptureController(make_gate(), "/tmp/test_capture")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    try:
        with patch('cv2.VideoCapture', return_value=mock_cap):
            ctrl._open_camera(99)
        print("✗ Phải raise RuntimeError")
    except RuntimeError as e:
        print(f"✓ RuntimeError khi không có camera local: {e}")

def test_open_rtsp_raises_if_unreachable():
    """Mock VideoCapture để không thật sự kết nối — tránh chờ timeout 50s."""
    ctrl = AutoCaptureController(make_gate(), "/tmp/test_capture")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    try:
        with patch('cv2.VideoCapture', return_value=mock_cap):
            ctrl._open_rtsp("rtsp://invalid:123@0.0.0.0:554/none")
        print("✗ Phải raise RuntimeError")
    except RuntimeError as e:
        print(f"✓ RuntimeError khi RTSP không kết nối được: {e}")

# ── Session tests ────────────────────────────────────────────────

def _session_patches(ctrl, mock_cap):
    """Patches dùng chung cho session tests — mock time để chạy nhanh."""
    fake_time = [0.0]
    def advance_time():
        fake_time[0] += 0.5  # mỗi lần gọi tăng 0.5s, luôn qua MIN_INTERVAL_SEC
        return fake_time[0]

    return [
        patch.object(ctrl, '_open_camera', return_value=mock_cap),
        patch('auto_capture.time.time', side_effect=advance_time),
        patch('cv2.imshow'),
        patch('cv2.waitKey', return_value=0),
        patch('cv2.destroyAllWindows'),
        patch('cv2.imwrite', return_value=True),
    ]

def test_session_saves_50_files(tmp_path):
    ctrl = AutoCaptureController(make_gate(), str(tmp_path))
    with patch.object(ctrl, '_open_camera', return_value=make_mock_cap()), \
         patch('auto_capture.time.time', side_effect=iter(i * 0.5 for i in range(10000))), \
         patch('cv2.imshow'), patch('cv2.waitKey', return_value=0), \
         patch('cv2.destroyAllWindows'), patch('cv2.imwrite', return_value=True):
        files = ctrl.run_session("product_001")

    assert len(files) == 50, f"Phải lưu 50 ảnh, thực tế: {len(files)}"
    print(f"✓ Session lưu đúng 50 ảnh")

def test_session_stops_on_camera_failure(tmp_path):
    ctrl = AutoCaptureController(make_gate(), str(tmp_path))
    with patch.object(ctrl, '_open_camera', return_value=make_mock_cap(fail_after=5)), \
         patch('auto_capture.time.time', side_effect=iter(i * 0.5 for i in range(10000))), \
         patch('cv2.imshow'), patch('cv2.waitKey', return_value=0), \
         patch('cv2.destroyAllWindows'), patch('cv2.imwrite', return_value=True):
        files = ctrl.run_session("product_002")

    assert len(files) < 50, "Session phải dừng sớm khi camera lỗi"
    print(f"✓ Session dừng khi camera lỗi ({len(files)} ảnh)")

def test_session_rtsp_mock(tmp_path):
    ctrl = AutoCaptureController(make_gate(), str(tmp_path))
    with patch.object(ctrl, '_open_camera', return_value=make_mock_cap()) as mock_open, \
         patch('auto_capture.time.time', side_effect=iter(i * 0.5 for i in range(10000))), \
         patch('cv2.imshow'), patch('cv2.waitKey', return_value=0), \
         patch('cv2.destroyAllWindows'), patch('cv2.imwrite', return_value=True):
        files = ctrl.run_session("product_rtsp", cam_source=RTSP_URL)

    mock_open.assert_called_once_with(RTSP_URL)
    assert len(files) == 50
    print(f"✓ Session RTSP mock chạy đúng: {len(files)} ảnh")

# ── Test thật với camera RTSP (cần mạng) ────────────────────────

def test_rtsp_connection_real():
    """
    Chạy test này chỉ khi camera RTSP đang online.
    Kiểm tra kết nối và đọc 5 frame.
    """
    print(f"\nKết nối RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("✗ Không kết nối được RTSP (camera offline?)")
        return

    ok_count = 0
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok_count += 1

    cap.release()
    assert ok_count == 5, f"Chỉ đọc được {ok_count}/5 frame"
    print(f"✓ RTSP hoạt động, đọc 5 frame thành công (shape: {frame.shape})")

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    print("=== Unit Tests ===")
    test_frame_hash_detects_similar()
    test_frame_hash_different()
    test_open_local_camera_raises_if_missing()
    test_open_rtsp_raises_if_unreachable()

    print("\n=== Session Tests (mock) ===")
    with tempfile.TemporaryDirectory() as tmp:
        test_session_saves_50_files(Path(tmp) / "a")
        test_session_stops_on_camera_failure(Path(tmp) / "b")
        test_session_rtsp_mock(Path(tmp) / "c")

    print("\n=== RTSP Real Connection ===")
    test_rtsp_connection_real()

    print("\nTất cả test xong!")
