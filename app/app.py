"""Streamlit web app: upload an image or a video and get REAL / FAKE, a confidence score and a Grad-CAM heatmap.

Run from the project root:
    py -m streamlit run app/app.py
"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from inference import DEFAULT_MODEL, DeepfakeDetector  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov"}

st.set_page_config(page_title="Deepfake & AI Content Detector", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def load_detector() -> DeepfakeDetector:
    return DeepfakeDetector(DEFAULT_MODEL)


def show_verdict(label: str, confidence: float, detail: str) -> None:
    color = "#c62828" if label == "FAKE" else "#2e7d32"
    meaning = "AI-generated / manipulated" if label == "FAKE" else "looks authentic"
    st.markdown(
        f"<div style='padding:1.1rem;border-radius:14px;background:{color}22;border:2px solid {color};"
        f"text-align:center'><div style='font-size:2.8rem;font-weight:800;color:{color};line-height:1.1'>{label}</div>"
        f"<div style='font-size:1rem;margin-top:.2rem'>{meaning}</div>"
        f"<div style='font-size:1.3rem;font-weight:600;margin-top:.5rem'>Confidence: {confidence * 100:.1f}%</div>"
        f"<div style='font-size:.9rem;opacity:.75'>{detail}</div></div>",
        unsafe_allow_html=True,
    )


def heatmap_help() -> None:
    st.caption("Heatmap: warm (red/yellow) areas are the regions that pushed the model towards the verdict "
               "'FAKE'. It is an explanation aid, not proof.")


# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold (P(fake) ≥ threshold → FAKE)", 0.05, 0.95, 0.50, 0.05)
skip_detect = st.sidebar.checkbox("My file is already a cropped face (skip face detection)", value=False)
with st.sidebar.expander("Video options", expanded=False):
    fps = st.slider("Frames analysed per second", 0.5, 5.0, 1.0, 0.5)
    max_frames = st.slider("Maximum frames", 5, 60, 30, 5)
    aggregate_label = st.radio("Combine frames by", ["Average probability", "Majority vote"])
with st.sidebar.expander("Advanced", expanded=False):
    margin = st.slider("Margin around detected face", 0.0, 0.8, 0.3, 0.05)
aggregate = "mean" if aggregate_label == "Average probability" else "vote"

# ----------------------------------------------------------------------------- main page
st.title("🔍 Deepfake & AI Content Detector")
st.write("Upload a **photo** or a **short video** of a face. The system finds the face, decides whether it is "
         "**REAL** or **FAKE**, and shows a heatmap of the regions that influenced the decision.")

if not Path(DEFAULT_MODEL).exists():
    st.error(f"Model file not found: `{DEFAULT_MODEL}`. Put `best_model.pth` in the `models` folder and reload.")
    st.stop()

uploaded = st.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded is not None:
    ext = Path(uploaded.name).suffix.lower()
    if skip_detect:
        st.warning("Face detection is switched OFF in the sidebar, so the whole picture is sent to the model. "
                   "Only use this for images that are already tight face crops. Untick it for normal photos.")
    try:
        detector = load_detector()

        # ------------------------------------------------------------- image
        if ext in IMAGE_EXTS:
            image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            with st.spinner("Analysing the image..."):
                res = detector.predict_image(image, threshold=threshold, margin=margin, skip_detect=skip_detect)
            if res["status"] != "ok":
                st.warning(res["message"])
                st.image(image, caption="Uploaded image", width=380)
            else:
                show_verdict(res["label"], res["confidence"], f"Probability of FAKE: {res['fake_probability']:.3f}")
                c1, c2, c3 = st.columns(3)
                c1.image(res.get("annotated", image), caption="Original (detected face boxed)", use_container_width=True)
                c2.image(res["face_crop"], caption="Face used by the model", use_container_width=True)
                c3.image(res["heatmap"], caption="Grad-CAM heatmap", use_container_width=True)
                heatmap_help()

        # ------------------------------------------------------------- video
        elif ext in VIDEO_EXTS:
            if uploaded.size > 200 * 1024 * 1024:
                st.error("This video is very large. Please upload a shorter clip (under 200 MB).")
                st.stop()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            st.video(uploaded)
            bar = st.progress(0.0, text="Starting...")
            try:
                res = detector.predict_video(
                    tmp_path, fps=fps, max_frames=max_frames, threshold=threshold, margin=margin,
                    skip_detect=skip_detect, aggregate=aggregate,
                    progress_cb=lambda frac, msg: bar.progress(frac, text=msg))
            finally:
                bar.empty()
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if res["status"] != "ok":
                st.warning(res["message"])
            else:
                if aggregate == "vote":
                    detail = f"{res['fake_frame_ratio'] * 100:.0f}% of frames were classified FAKE"
                else:
                    detail = f"Average probability of FAKE over frames: {res['mean_fake_probability']:.3f}"
                show_verdict(res["label"], res["confidence"],
                             f"{detail} · {res['frames_analyzed']} frames analysed")
                st.subheader("Fake probability across the video")
                df = pd.DataFrame(res["per_frame"]).rename(
                    columns={"time_sec": "Time (s)", "fake_probability": "P(fake)"}).set_index("Time (s)")
                st.line_chart(df["P(fake)"], height=240)
                top = res["top_frame"]
                st.subheader(f"Most suspicious frame (at {top['time_sec']} s, P(fake) = {top['fake_probability']:.3f})")
                c1, c2 = st.columns(2)
                c1.image(res["top_crop"], caption="Face used by the model", use_container_width=True)
                c2.image(res["heatmap"], caption="Grad-CAM heatmap", use_container_width=True)
                heatmap_help()
        else:
            st.error("Unsupported file type. Please upload a JPG, PNG, MP4, AVI or MOV file.")
    except Exception as exc:  # keep the app friendly instead of showing a traceback
        st.error(f"Something went wrong while analysing this file: {exc}")

if uploaded is not None:
    try:
        _d = load_detector()
        if not skip_detect:
            st.caption(f"Face detector used: {_d.backend}" + (f"  \n(details: {_d.backend_note})" if _d.backend_note else ""))
    except Exception:
        pass

st.divider()
st.caption("Academic prototype. Results are probabilistic and not legal proof. "
           "The model was trained on real photos vs AI-generated faces and may be less reliable on other "
           "kinds of manipulation (e.g. face-swap videos), low-quality or partly hidden faces.")
