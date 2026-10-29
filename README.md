# Multimodal Attendance & Engagement Agent

A beginner-friendly, local prototype for the SmartBridge capstone project.

## What it currently does
- Detects faces in webcam frames or a supplied video using OpenCV's Haar cascade.
- Tracks face detections by approximate bounding-box overlap and records observed presence duration.
- Analyzes a WAV audio file using short-window RMS energy as a *speech/activity proxy* (not speech recognition).
- Combines visual presence and audio activity into a transparent rule-based session score.
- Generates simple teaching suggestions and CSV/JSON outputs.
- Offers an optional Streamlit dashboard.

## Privacy and limitations
- Runs locally; it does not upload webcam/video/audio to a server.
- It does not identify who a person is, perform biometric identity matching, or prove that someone is attentive.
- Face counts and box tracking can be inaccurate with occlusion, lighting changes, or camera movement.
- Audio energy cannot reliably distinguish speech from noise. Use consented recordings and follow your institution's privacy rules.
- Treat engagement scores as experimental indicators, not judgments about individual students.

## Setup (Windows / VS Code)
1. Install Python 3.10 or later and Git.
2. Open this folder in VS Code.
3. Open Terminal in VS Code and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt in VS Code and run:
```bat
.venv\Scripts\activate.bat
```

## Run the core app
```powershell
python main.py
```
Choose webcam analysis, video analysis, audio-only analysis, or sample/demo mode from the menu. Webcam access may require permission.

## Run the dashboard
```powershell
streamlit run dashboard.py
```

## Outputs
The `outputs/` folder is created automatically. Reports include:
- `attendance.csv`
- `audio_activity.csv`
- `session_report.json`

## Suggested GitHub repository description
A privacy-conscious local prototype combining OpenCV face detection, WAV audio activity analysis, rule-based engagement indicators, session analytics, and instructor recommendations.

## Demo checklist
1. Run `python main.py` and choose sample/demo mode.
2. Run `streamlit run dashboard.py` for the optional dashboard.
3. Capture screenshots of the running app and report.
4. Push the complete project folder to GitHub.
5. Only submit the actual repository/demo URLs in Skill Wallet after verifying they open.
