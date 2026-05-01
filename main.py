"""
main.py — Chạy trên Jetson qua SSH hoặc màn hình
Công nhân quét mã vạch hoặc nhập tên → hệ thống tự chụp 50 ảnh
"""
import os
os.environ.setdefault("DISPLAY", ":0")

from dotenv import load_dotenv
load_dotenv()

from quality_gate import JetsonQualityGate
from auto_capture import AutoCaptureController

CAM_SOURCE = int(os.getenv("CAM_INDEX", 0))
SAVE_DIR   = os.getenv("SAVE_DIR", "/home/haiprotn/Documents/detect_product_ai/data/products")
SHOW_UI    = True

def main():
    gate = JetsonQualityGate()
    ctrl = AutoCaptureController(gate, save_dir=SAVE_DIR)

    print("=" * 45)
    print("  Hệ thống chụp ảnh sản phẩm — Sẵn sàng")
    print("=" * 45)
    print(f"  Lưu ảnh vào: {SAVE_DIR}")
    print(f"  Camera:      USB index={CAM_SOURCE}")
    print("  Gõ 'q' để thoát")
    print("=" * 45)

    while True:
        try:
            product_id = input("\nQuét mã vạch hoặc nhập tên sản phẩm: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if product_id.lower() == "q":
            print("Thoát.")
            break
        if not product_id:
            continue

        print(f"\nBắt đầu chụp: [{product_id}]")

        files = ctrl.run_session(
            product_id=product_id,
            cam_source=CAM_SOURCE,
            show_ui=SHOW_UI,
        )

        print(f"Xong! Đã lưu {len(files)} ảnh vào: {SAVE_DIR}/{product_id}/")
        print("-" * 45)

if __name__ == "__main__":
    main()
