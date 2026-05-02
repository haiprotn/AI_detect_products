"""
recognize.py — Test nhận dạng cục bộ (không qua server)
Dùng camera USB hoặc RTSP, load DINOv2 + .pkl trực tiếp.
"""
import os
import pickle
import threading
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import deque

warnings.filterwarnings("ignore")

VECTOR_DIR  = Path(os.getenv("VECTOR_DIR", "./vector_store"))
CAM_SOURCE  = os.getenv("RTSP_URL") or int(os.getenv("CAM_INDEX", 0))
MODEL_NAME  = "dinov2_vitb14"   # base model — 768 dim
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
INTERVAL    = 1.0
CROP_RATIO  = 0.60
IMG_SIZE    = 224
VOTE_WIN    = 7
SCORE_FLOOR = 0.65
MARGIN      = 0.02
DISPLAY_W   = 960
DISPLAY_H   = 540

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    print(f"Loading {MODEL_NAME}...")
    m = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    m.eval().to(DEVICE)
    print(f"Model ready | device={DEVICE}")
    return m


def load_db():
    pkls = sorted(VECTOR_DIR.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"Không có .pkl trong {VECTOR_DIR}")
    db = []
    for p in pkls:
        with open(p, "rb") as f:
            db.append(pickle.load(f))
        print(f"  {db[-1]['product_id']}: {db[-1]['n_images']} vectors")
    return db


def embed(model, frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t   = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = model(t).cpu().numpy().flatten()
    v = v / np.linalg.norm(v)
    return v.astype(np.float32)


def search(db, vec, top_k=3):
    results = []
    for d in db:
        sims  = d["vectors"] @ vec
        score = float(np.sort(sims)[::-1][:3].mean())
        results.append({"product_id": d["product_id"],
                        "category": d["category"], "score": round(score, 4)})
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


def crop_box(h, w):
    side = int(min(h, w) * CROP_RATIO)
    cx, cy = w//2, h//2
    return cx-side//2, cy-side//2, cx+side//2, cy+side//2


class FrameGrabber(threading.Thread):
    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.frame  = None
        self._lock  = threading.Lock()
        self._stop  = False

    def run(self):
        if isinstance(self.source, str):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|threads;1"
            )
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not self._stop:
            try:
                ok, f = cap.read()
                if ok:
                    with self._lock:
                        self.frame = f
                else:
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.1)
        cap.release()

    def latest(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stop = True


def draw(frame, results, elapsed, voted):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = crop_box(h, w)

    blur_bg = cv2.GaussianBlur(frame, (31, 31), 0)
    mask    = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = 255
    out = np.where(mask[:,:,None] == 255, frame, blur_bg)

    # Khung
    cv2.rectangle(out, (x1,y1), (x2,y2), (200,200,200), 1)

    # Header
    cv2.rectangle(out, (0,0), (w,52), (0,0,0), -1)
    cv2.putText(out, f"DINOv2  {elapsed:.0f}ms",
                (12,34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160,160,160), 1)

    # Kết quả
    if voted and results:
        top = next((r for r in results if r["product_id"]==voted), results[0])
        gap = results[0]["score"] - results[1]["score"] if len(results)>1 else 1.0
        if top["score"] >= SCORE_FLOOR and gap >= MARGIN:
            label = f">> {top['product_id']}  {top['score']:.3f}"
            color = (0, 255, 80)
        else:
            label = f"??  {top['product_id']}  score={top['score']:.3f}"
            color = (0, 165, 255)
        cv2.putText(out, label, (12,90),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2)

    for i, r in enumerate(results):
        c = (100,255,100) if i==0 else (150,150,150)
        cv2.putText(out, f"  #{i+1} {r['product_id']}  {r['score']:.4f}",
                    (12, 120+i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

    return cv2.resize(out, (DISPLAY_W, DISPLAY_H))


def main():
    model    = load_model()
    db       = load_db()
    grabber  = FrameGrabber(CAM_SOURCE)
    grabber.start()

    print("Q=thoat | Space=nhan dang ngay")
    while grabber.latest() is None:
        time.sleep(0.05)

    results, elapsed, last_time = [], 0.0, 0.0
    vote_buf = deque(maxlen=VOTE_WIN)

    while True:
        frame = grabber.latest()
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        if now - last_time >= INTERVAL:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = crop_box(h, w)
            roi   = frame[y1:y2, x1:x2]
            t0    = time.time()
            vec   = embed(model, roi)
            results = search(db, vec)
            elapsed = (time.time()-t0)*1000
            last_time = now
            if results:
                vote_buf.append(results[0]["product_id"])
                print(f"Top1: {results[0]['product_id']}  "
                      f"score={results[0]['score']:.4f}  {elapsed:.0f}ms")

        voted = (max(set(vote_buf), key=list(vote_buf).count)
                 if vote_buf else None)

        cv2.imshow("Nhan dang  [Q|Space]", draw(frame, results, elapsed, voted))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            last_time = 0

    grabber.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
