import streamlit as st
import cv2
import pytesseract
import re
import numpy as np
import pandas as pd
from functools import lru_cache

# ---------- Configuration ----------
VIDEO_PATH = "history_game.mp4"  # place your uploaded MP4 here
FRAME_INTERVAL = 150  # extract every 150th frame (~5 seconds for 30fps)
ROI = None  # (x, y, w, h) or None to use full frame

# ---------- Utility Functions ----------
@st.cache_resource
@lru_cache(maxsize=1)
def extract_multipliers(video_path):
    """
    Extracts multiplier history from video using OCR.
    Returns a list of floats.
    """
    cap = cv2.VideoCapture(video_path)
    multipliers = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % FRAME_INTERVAL == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ROI:
                x, y, w, h = ROI
                gray = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(gray, config='--psm 6 digits')
            found = re.findall(r"\d+\.\d+", text)
            for f in found:
                try:
                    multipliers.append(float(f))
                except:
                    pass
        count += 1
    cap.release()
    return multipliers

# ---------- Prediction Model ----------
def compute_confidence(data, threshold=2.0):
    """
    Given a list of multipliers, returns probability for above and under threshold.
    """
    if not data:
        return 0.5, 0.5
    data = np.array(data)
    above = np.sum(data > threshold)
    under = np.sum(data <= threshold)
    total = above + under
    return above/total, under/total

# ---------- Streamlit App ----------
def main():
    st.title("Crash Game Predictor")

    # Load history
    st.sidebar.header("Data Source")
    use_video = st.sidebar.checkbox("Load from video", value=True)
    data = []
    if use_video:
        st.sidebar.write("Reading multipliers from video...")
        data = extract_multipliers(VIDEO_PATH)
        st.sidebar.success(f"Extracted {len(data)} multipliers")

    # Manual entry
    st.sidebar.header("Manual Input")
    new_val = st.sidebar.text_input("Enter new multiplier (e.g., 1.87)")
    if st.sidebar.button("Add to history"):
        try:
            f = float(new_val)
            data.append(f)
            st.sidebar.success(f"Added {f} to history")
        except:
            st.sidebar.error("Invalid number format.")

    # Display recent history
    st.subheader("Recent History (last 10)")
    if data:
        recent = data[-10:]
        st.write(recent)
    else:
        st.write("No data available.")

    # Compute confidence
    above_conf, under_conf = compute_confidence(data)
    st.subheader("Prediction Confidence")
    st.write(f"Above 200%: {above_conf:.1%}")
    st.write(f"Under 200%: {under_conf:.1%}")

    # Prediction buttons
    st.subheader("Make a Prediction")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict Above 2"):
            st.write(f"Prediction: Above 200% ({above_conf:.1%} confidence)")
    with col2:
        if st.button("Predict Under 2"):
            st.write(f"Prediction: Under 200% ({under_conf:.1%} confidence)")

    # Footer
    st.markdown("---")
    st.write("*Built with Streamlit — add your MP4 file named 'history_game.mp4' to the app directory.*")

if __name__ == "__main__":
    main()
