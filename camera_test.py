# camera_test.py — Nhận dạng cục bộ với DINOv2 + .pkl
import argparse
import pickle
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

CAM_INDEX   = 0
VECTOR_DIR  = Path("./vector_store")
MODEL_NAME  = "dinov2_vits14"
SCORE_FLOOR  = 0.75   # dưới mức này = không phải sản phẩm đã học
MARGIN_MIN   = 0.015  # top1 phải cao hơn top2 ít nhất mức này
VOTE_WINDOW  = 9
YOLO_CONF    = 0.40   # confidence YOLO — cao hơn để lọc nhiễu nền
BOX_MIN_AREA = 0.05   # bbox phải chiếm ít nhất 5% diện tích frame
INTERVAL    = 0.8     # giây giữa 2 lần nhận dạng
DISPLAY_W   = 960
DISPLAY_H   = 540
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Load model ────────────────────────────────────────────────────────────────

def load_model():
    import warnings
    warnings.filterwarnings("ignore", message="xFormers is not available")
    print(f"[DINOv2] Loading {MODEL_NAME}...")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval().to(DEVICE)
    print(f"[DINOv2] Ready | device={DEVICE}")
    return model


def load_yolo():
    print("[YOLO] Loading yolov8n.pt...")
    yolo = YOLO("yolov8n.pt")
    print("[YOLO] Ready")
    return yolo


def crop_product(yolo, frame: np.ndarray, padding: float = 0.20):
    """Detect + crop sản phẩm. Trả về (cropped_frame, has_object)."""
    h, w = frame.shape[:2]
    results = yolo(frame, verbose=False, conf=YOLO_CONF)
    boxes   = results[0].boxes

    if len(boxes) == 0:
        return None, False

    xywh    = boxes.xywh.cpu().numpy()
    largest = xywh[np.argmax(xywh[:, 2] * xywh[:, 3])]
    cx, cy, bw, bh = largest

    # Box phải chiếm ít nhất BOX_MIN_AREA của frame
    if (bw * bh) < (w * h * BOX_MIN_AREA):
        return None, False

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, int(cx - bw/2) - pad_x)
    y1 = max(0, int(cy - bh/2) - pad_y)
    x2 = min(w, int(cx + bw/2) + pad_x)
    y2 = min(h, int(cy + bh/2) + pad_y)

    return frame[y1:y2, x1:x2], True


# ── Load .pkl database ────────────────────────────────────────────────────────

def load_db() -> list[dict]:
    pkls = sorted(VECTOR_DIR.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"Không có .pkl trong {VECTOR_DIR} — build vector trước")
    db = []
    for p in pkls:
        with open(p, "rb") as f:
            data = pickle.load(f)
        db.append(data)
        print(f"[DB] {data['product_id']}: {data['n_images']} ảnh")
    return db


# ── Embed 1 frame ─────────────────────────────────────────────────────────────

def embed(model, frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t   = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vec = model(t).cpu().numpy().flatten()
    vec = vec / np.linalg.norm(vec)
    return vec.astype(np.float32)


# ── Tìm kiếm trong .pkl ───────────────────────────────────────────────────────

def search(db: list[dict], vec: np.ndarray, top_k: int = 5) -> list[dict]:
    results = []
    for data in db:
        # cosine similarity với tất cả vectors của sản phẩm → lấy trung bình top-3
        sims   = data["vectors"] @ vec          # (N,)
        top3   = np.sort(sims)[::-1][:3]
        score  = float(top3.mean())
        results.append({
            "product_id": data["product_id"],
            "category":   data["category"],
            "score":      round(score, 4),
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


# ── RTSP/USB reader thread ────────────────────────────────────────────────────

class CameraReader(threading.Thread):
    def __init__(self, index: int):
        super().__init__(daemon=True)
        self.index = index
        self.frame = None
        self.lock  = threading.Lock()
        self._stop = False

    def run(self):
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[Lỗi] Không mở được camera {self.index}")
            return
        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame
        cap.release()

    def latest(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stop = True


# ── Vẽ kết quả ───────────────────────────────────────────────────────────────

def draw(frame: np.ndarray, results: list[dict], elapsed_ms: float,
         voted: str | None, floor: float = SCORE_FLOOR, margin: float = MARGIN_MIN):
    n     = max(1, len(results))
    box_h = 90 + 30 * n
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], box_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"DINOv2  ({elapsed_ms:.0f} ms)",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    if voted and results:
        top  = next((r for r in results if r["product_id"] == voted), results[0])
        gap  = results[0]["score"] - results[1]["score"] if len(results) > 1 else 1.0
        confident = top["score"] >= floor and gap >= margin
        if confident:
            label = f">> {top['product_id']}  ({top['score']:.3f}  +{gap:.3f})"
            color = (0, 255, 80)
        else:
            label = f"?? Khong chac  score={top['score']:.3f}  gap={gap:.3f}"
            color = (0, 165, 255)
        cv2.putText(frame, label, (12, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)
    else:
        cv2.putText(frame, "Dang nhan dang...",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

    for i, r in enumerate(results):
        color = (100, 255, 100) if i == 0 else (150, 150, 150)
        cv2.putText(frame,
                    f"  #{i+1} {r['product_id']}  {r['score']:.4f}",
                    (12, 90 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam",      type=int,   default=CAM_INDEX)
    parser.add_argument("--width",    type=int,   default=DISPLAY_W)
    parser.add_argument("--height",   type=int,   default=DISPLAY_H)
    parser.add_argument("--interval", type=float, default=INTERVAL)
    parser.add_argument("--floor",    type=float, default=SCORE_FLOOR)
    parser.add_argument("--margin",   type=float, default=MARGIN_MIN)
    args = parser.parse_args()

    floor  = args.floor
    margin = args.margin

    model    = load_model()
    yolo     = load_yolo()
    db       = load_db()
    reader   = CameraReader(args.cam)
    reader.start()

    print(f"Camera: {args.cam} | Floor: {floor} | Margin: {margin} | Q=thoat | Space=ngay")

    results   : list[dict] = []
    elapsed_ms: float      = 0.0
    last_time : float      = 0.0
    vote_buf               = deque(maxlen=VOTE_WINDOW)

    while reader.latest() is None:
        time.sleep(0.05)

    while True:
        frame = reader.latest()
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        if now - last_time >= args.interval:
            t0 = time.time()
            cropped, has_obj = crop_product(yolo, frame)
            if has_obj:
                vec     = embed(model, cropped)
                hits    = search(db, vec)
                elapsed_ms = (time.time() - t0) * 1000
                last_time  = now
                # Lọc score floor — dưới ngưỡng không tính
                if hits and hits[0]["score"] >= floor:
                    results = hits
                    vote_buf.append(results[0]["product_id"])
                    print(f"Top1: {results[0]['product_id']}  "
                          f"score={results[0]['score']:.4f}  {elapsed_ms:.0f}ms")
                else:
                    results = []
                    s = hits[0]['score'] if hits else 0
                    print(f"Score thấp ({s:.4f} < {floor}) — bỏ qua")
            else:
                results    = []
                elapsed_ms = 0.0
                last_time  = now
                print("Không thấy sản phẩm")

        voted = max(set(vote_buf), key=list(vote_buf).count) if vote_buf else None

        display = cv2.resize(frame, (args.width, args.height))
        draw(display, results, elapsed_ms, voted, floor, margin)
        cv2.imshow("Nhan dang san pham  [Q | Space]", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            last_time = 0

    reader.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
