# Crash Game Predictor

This Streamlit app predicts whether the next multiplier in a crash game will be above or under 2.0 (200%).

## Features
- Extracts historical multipliers from a video file via OCR.
- Frequency-based prediction with confidence scores.
- Displays recent history (last 10 values).
- Manual input to update history.
- Two prediction buttons: **Above 2** and **Under 2**.

## Setup
1. Rename your crash game video to `history_game.mp4` and place it in the app directory.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Deployment
- Push this repository to GitHub.
- Connect to Streamlit Cloud for hosting.
