"""
main.py — Chạy trên Jetson qua SSH hoặc màn hình
Công nhân quét mã vạch hoặc nhập tên → hệ thống tự chụp 50 ảnh → tự upload
"""
import os
os.environ.setdefault("DISPLAY", ":0")

from dotenv import load_dotenv
load_dotenv()

from quality_gate import JetsonQualityGate
from auto_capture import AutoCaptureController
from transfer_engine import TransferEngine

CAM_SOURCE = int(os.getenv("CAM_INDEX", 0))
SAVE_DIR   = os.getenv("SAVE_DIR", "/home/haiprotn/Documents/detect_product_ai/data/products")
SHOW_UI    = True
CATEGORY   = os.getenv("CATEGORY", "san_pham")  # nhóm sản phẩm trên server

def main():
    gate     = JetsonQualityGate()
    ctrl     = AutoCaptureController(gate, save_dir=SAVE_DIR)
    transfer = TransferEngine()

    print("=" * 45)
    print("  Hệ thống chụp ảnh sản phẩm — Sẵn sàng")
    print("=" * 45)
    print(f"  Lưu ảnh vào: {SAVE_DIR}")
    print(f"  Camera:      USB index={CAM_SOURCE}")
    print(f"  Server:      {TransferEngine.SERVER_URL}")
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

        print(f"Xong! Đã lưu {len(files)} ảnh")

        if files:
            job_id = transfer.enqueue(files, product_id, category=CATEGORY)
            print(f"Đã đưa vào hàng đợi upload (job: {job_id})")
            print(f"→ Chạy 'python transfer_engine.py' để upload lên server")

        print("-" * 45)

if __name__ == "__main__":
    main()
