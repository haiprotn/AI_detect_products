"""
preview.py — Xem camera + tự động chụp khi xoay sản phẩm
  S = bắt đầu/dừng chụp
  C = xóa ảnh đã chụp trong session này
  Q = thoát
"""
import os
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ.setdefault("DISPLAY", ":0")

from dotenv import load_dotenv
load_dotenv()

CAM_INDEX = 0
SAVE_DIR  = os.getenv("SAVE_DIR", "/home/haiprotn/Documents/detect_product_ai/data/products")

# Chụp khi góc nhìn thay đổi >= ngưỡng này (0-64 bits pHash)
ANGLE_CHANGE_THRESH = 8   # tăng = ít nhạy hơn, giảm = nhạy hơn
MIN_CAPTURE_SEC     = 0.8 # giây tối thiểu giữa 2 lần chụp
MAX_CAPTURES        = 50

SCREEN_W, SCREEN_H = 1366, 768


def phash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (8, 8)).astype(float)
    return (small.flatten() > small.mean()).astype(np.uint8)


def hash_diff(h1, h2) -> int:
    return int(np.sum(h1 != h2))  # 0-64, thấp = giống nhau


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Không mở được camera index={CAM_INDEX}")
        return

    # Tạo thư mục session
    session_id  = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    product_id  = input("Tên sản phẩm (Enter để dùng tên session): ").strip()
    if not product_id:
        product_id = session_id
    save_path = Path(SAVE_DIR) / product_id
    save_path.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture", SCREEN_W, SCREEN_H)

    capturing       = False
    captured        = 0
    last_hash       = None
    last_cap_time   = 0
    just_captured   = 0.0   # timestamp flash effect

    print(f"\nLưu vào: {save_path}")
    print("S = bắt đầu chụp | Q = thoát\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_hash = phash(gray)

        # ── Auto capture khi góc thay đổi ──────────────────
        if capturing and captured < MAX_CAPTURES:
            now = time.time()
            diff = hash_diff(cur_hash, last_hash) if last_hash is not None else 99
            interval_ok = (now - last_cap_time) >= MIN_CAPTURE_SEC

            if diff >= ANGLE_CHANGE_THRESH and interval_ok:
                fname = save_path / f"{product_id}_{captured:03d}.jpg"
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                captured      += 1
                last_cap_time  = now
                last_hash      = cur_hash
                just_captured  = now
                print(f"  [{captured:02d}/{MAX_CAPTURES}] {fname.name}")

            if captured >= MAX_CAPTURES:
                capturing = False
                print(f"\nHoàn thành! {captured} ảnh lưu tại {save_path}")

        elif last_hash is None:
            last_hash = cur_hash

        # ── Vẽ UI ───────────────────────────────────────────
        display = cv2.resize(frame, (SCREEN_W, SCREEN_H))
        h, w    = display.shape[:2]

        # Flash xanh khi vừa chụp
        if time.time() - just_captured < 0.15:
            cv2.rectangle(display, (0, 0), (w, h), (0, 255, 0), 12)

        # Khung sản phẩm
        box_color = (0, 255, 0) if capturing else (0, 200, 255)
        cv2.rectangle(display,
                      (int(w * 0.15), int(h * 0.1)),
                      (int(w * 0.85), int(h * 0.9)),
                      box_color, 2)

        # Thanh tiến trình
        if capturing or captured > 0:
            prog_w = int(w * 0.7 * captured / MAX_CAPTURES)
            cv2.rectangle(display, (int(w*0.15), h-30), (int(w*0.85), h-10), (50,50,50), -1)
            cv2.rectangle(display, (int(w*0.15), h-30), (int(w*0.15)+prog_w, h-10), (0,220,0), -1)

        # Header
        cv2.rectangle(display, (0, 0), (w, 48), (0, 0, 0), -1)
        status = "DANG CHUP — xoay san pham cham" if capturing else "S = bat dau chup  |  Q = thoat"
        cv2.putText(display, status, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if capturing else (0, 200, 255), 2)

        # Số ảnh
        cv2.putText(display, f"{captured}/{MAX_CAPTURES}",
                    (w - 160, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Capture", display)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            capturing = not capturing
            if capturing:
                last_hash     = cur_hash
                last_cap_time = 0
                print(f"Bắt đầu chụp — xoay sản phẩm từ từ...")
            else:
                print(f"Tạm dừng. Đã chụp {captured} ảnh.")
        elif key == ord('c'):
            for f in save_path.glob("*.jpg"):
                f.unlink()
            captured  = 0
            last_hash = None
            print("Đã xóa ảnh trong session này.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nKết thúc. Tổng: {captured} ảnh tại {save_path}")


if __name__ == "__main__":
    main()
