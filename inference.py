"""Inference for the Deepfake and AI Content Detector.

* MTCNN finds and crops the face (skipped for images that are already cropped faces).
* EfficientNet-B0 (models/best_model.pth) predicts P(fake).
* Grad-CAM shows which regions pushed the model towards "FAKE".

Quick command-line test (from the project root):
    py src\\inference.py path\\to\\image.jpg
    py src\\inference.py path\\to\\video.mp4
    py src\\inference.py path\\to\\cropped_face.jpg --skip_detect
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

try:  # works for "py src\inference.py", "py -m src.inference" and imports from the app
    from src.model import build_model
except ImportError:
    from model import build_model

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "best_model.pth"
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

ImageLike = Union[str, Path, Image.Image]


class _FakeLogitTarget:
    """Grad-CAM target: explain the single 'fake' logit (high logit = FAKE)."""

    def __call__(self, model_output):
        return model_output.reshape(-1)[0]


class DeepfakeDetector:
    """Loads the trained model once and offers predict_image() / predict_video()."""

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL, device: Optional[str] = None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(pretrained=False)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self._mtcnn = None  # created on first use, so skip_detect works without facenet-pytorch

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load(image: ImageLike) -> Image.Image:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        return ImageOps.exif_transpose(img).convert("RGB")

    def _get_mtcnn(self):
        """MTCNN from facenet-pytorch. If the package is missing it is installed once without its
        old torch pin; if that fails too, returns False and OpenCV's Haar detector is used instead."""
        if self._mtcnn is None:
            MTCNN = None
            try:
                from facenet_pytorch import MTCNN
            except ImportError:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q", "facenet-pytorch"],
                                   check=True, timeout=300)
                    importlib.invalidate_caches()
                    from facenet_pytorch import MTCNN
                except Exception:
                    MTCNN = None
            self._mtcnn = MTCNN(keep_all=True, device="cpu", post_process=False) if MTCNN else False
        return self._mtcnn

    @staticmethod
    def _haar_detect(img: Image.Image):
        """Fallback face detector (OpenCV, frontal faces). Returns (boxes, probs) like MTCNN."""
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(found) == 0:
            return None, None
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in found], dtype=float)
        return boxes, [1.0] * len(boxes)

    def detect_face(self, img: Image.Image, margin: float = 0.3):
        """Return (square face crop, (x1, y1, x2, y2)) for the largest face, or None."""
        mtcnn = self._get_mtcnn()
        boxes, probs = mtcnn.detect(img) if mtcnn else self._haar_detect(img)
        if boxes is None or len(boxes) == 0:
            return None
        best, best_area = None, 0.0
        for box, p in zip(boxes, probs):
            if p is None or p < 0.90:
                continue
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            if area > best_area:
                best, best_area = box, area
        if best is None:
            return None
        x1, y1, x2, y2 = [float(v) for v in best]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        W, H = img.size
        side = min(max(x2 - x1, y2 - y1) * (1 + 2 * margin), W, H)
        left = min(max(cx - side / 2, 0), W - side)
        top = min(max(cy - side / 2, 0), H - side)
        box_int = (int(left), int(top), int(left + side), int(top + side))
        return img.crop(box_int), box_int

    def _prob(self, crop: Image.Image) -> float:
        x = self.tf(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(x)).item())

    def explain(self, crop: Image.Image) -> np.ndarray:
        """Grad-CAM heatmap overlaid on the face crop (uint8 RGB array)."""
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        x = self.tf(crop).unsqueeze(0).to(self.device)
        rgb = np.asarray(crop.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        with torch.enable_grad():
            with GradCAM(model=self.model, target_layers=[self.model.features[-1]]) as cam:
                gray = cam(input_tensor=x, targets=[_FakeLogitTarget()])[0]
        return show_cam_on_image(rgb, gray, use_rgb=True)

    # ------------------------------------------------------------------ images
    def predict_image(self, image: ImageLike, threshold: float = 0.5, margin: float = 0.3,
                      skip_detect: bool = False, explain: bool = True) -> dict:
        """Classify one image. Returns a dict; check result['status'] first ('ok' or 'no_face')."""
        img = self._load(image)
        box = None
        if skip_detect:
            crop = img.copy()
        else:
            found = self.detect_face(img, margin)
            if found is None:
                return {"status": "no_face",
                        "message": "No face found in this image. Try a clearer, front-facing photo."}
            crop, box = found
        prob = self._prob(crop)
        label = "FAKE" if prob >= threshold else "REAL"
        out = {
            "status": "ok",
            "label": label,
            "fake_probability": prob,
            "confidence": prob if label == "FAKE" else 1.0 - prob,
            "threshold": threshold,
            "box": box,
            "face_crop": crop.resize((IMG_SIZE, IMG_SIZE)),
        }
        if box is not None:
            annotated = img.copy()
            ImageDraw.Draw(annotated).rectangle(box, outline=(255, 0, 0), width=max(2, img.size[0] // 200))
            out["annotated"] = annotated
        if explain:
            out["heatmap"] = self.explain(crop)
        return out

    # ------------------------------------------------------------------ videos
    def predict_video(self, path: Union[str, Path], fps: float = 1.0, max_frames: int = 30,
                      threshold: float = 0.5, margin: float = 0.3, skip_detect: bool = False,
                      aggregate: str = "mean", explain: bool = True,
                      progress_cb: Optional[Callable[[float, str], None]] = None) -> dict:
        """Classify a video by sampling frames (about `fps` per second, at most `max_frames`).

        aggregate='mean': average the per-frame fake probabilities.
        aggregate='vote': majority vote over frames.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {"status": "error", "message": "Could not open this video file."}
        vfps = cap.get(cv2.CAP_PROP_FPS)
        if not vfps or vfps != vfps or vfps <= 0:
            vfps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(vfps / max(fps, 0.01))))

        wanted = None
        if total > 0:
            idxs = list(range(0, total, step))
            if len(idxs) > max_frames:
                idxs = sorted({idxs[int(i)] for i in np.linspace(0, len(idxs) - 1, max_frames)})
            wanted = set(idxs)
        n_target = len(wanted) if wanted else max_frames

        records, best = [], None  # best = (prob, crop)
        i = sampled = 0
        while True:
            if not cap.grab():
                break
            take = (i in wanted) if wanted is not None else (i % step == 0 and sampled < max_frames)
            if take:
                ok, frame = cap.retrieve()
                sampled += 1
                if ok:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    crop = img if skip_detect else (self.detect_face(img, margin) or (None, None))[0]
                    if crop is not None:
                        p = self._prob(crop)
                        records.append({"frame": i, "time_sec": round(i / vfps, 2), "fake_probability": p})
                        if best is None or p > best[0]:
                            best = (p, crop)
                if progress_cb:
                    progress_cb(min(sampled / max(n_target, 1), 1.0), f"Analysed {sampled}/{n_target} frames")
            i += 1
            if wanted is not None and i > max(wanted, default=0):
                break
            if wanted is None and sampled >= max_frames:
                break
        cap.release()

        if not records:
            return {"status": "no_face", "frames_sampled": sampled,
                    "message": "No face found in the sampled frames of this video."}

        probs = np.array([r["fake_probability"] for r in records])
        if aggregate == "vote":
            ratio = float((probs >= threshold).mean())
            label = "FAKE" if ratio > 0.5 else "REAL"
            score, confidence = ratio, max(ratio, 1.0 - ratio)
        else:
            score = float(probs.mean())
            label = "FAKE" if score >= threshold else "REAL"
            confidence = score if label == "FAKE" else 1.0 - score

        out = {
            "status": "ok",
            "label": label,
            "score": score,
            "confidence": confidence,
            "aggregate": aggregate,
            "threshold": threshold,
            "mean_fake_probability": float(probs.mean()),
            "fake_frame_ratio": float((probs >= threshold).mean()),
            "frames_sampled": sampled,
            "frames_analyzed": len(records),
            "per_frame": records,
            "top_frame": max(records, key=lambda r: r["fake_probability"]),
            "top_crop": best[1].resize((IMG_SIZE, IMG_SIZE)),
        }
        if explain:
            out["heatmap"] = self.explain(best[1])
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify one image or video as REAL or FAKE.")
    ap.add_argument("path")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--skip_detect", action="store_true", help="input is already a cropped face")
    args = ap.parse_args()

    det = DeepfakeDetector(args.model)
    is_video = Path(args.path).suffix.lower() in VIDEO_EXTS
    res = (det.predict_video(args.path, threshold=args.threshold, skip_detect=args.skip_detect)
           if is_video else
           det.predict_image(args.path, threshold=args.threshold, skip_detect=args.skip_detect))
    if res["status"] != "ok":
        print(res["message"])
        return
    print(f"Verdict   : {res['label']}")
    print(f"Confidence: {res['confidence'] * 100:.1f}%")
    if is_video:
        print(f"Frames    : {res['frames_analyzed']} analysed, mean fake prob {res['mean_fake_probability']:.3f}")
    else:
        print(f"P(fake)   : {res['fake_probability']:.3f}")
    out = ROOT / "results" / "last_heatmap.png"
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(res["heatmap"]).save(out)
    print(f"Heatmap saved to {out}")


if __name__ == "__main__":
    main()
