import csv
import json
import os
import time
import wave
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def detect_video(source=0, show_window=True, min_presence_seconds=3.0):
    """Detect faces and approximate continuous presence; does not identify people."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        cap.release()
        raise RuntimeError("OpenCV face detector could not be loaded.")
    started = time.time()
    observations = []
    total_frames = 0
    face_frames = 0
    print("Press Q to stop the video window.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
            now = time.time()
            if len(boxes):
                face_frames += 1
            for i, (x, y, w, h) in enumerate(boxes, start=1):
                observations.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_seconds": round(now - started, 2),
                    "detection_number_in_frame": i,
                    "x": int(x), "y": int(y), "width": int(w), "height": int(h)
                })
                cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 80), 2)
            cv2.putText(frame, f"Faces detected: {len(boxes)} | Q to stop",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 80), 2)
            if show_window:
                cv2.imshow("Multimodal Attendance Agent - local prototype", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    elapsed = max(time.time() - started, 0.1)
    face_frame_ratio = face_frames / max(total_frames, 1)
    # Prototype-level session presence indicator, not individual attendance identity.
    attendance_row = {
        "session_date": datetime.now().date().isoformat(),
        "source": "webcam" if source == 0 else str(source),
        "duration_seconds": round(elapsed, 2),
        "frames_processed": total_frames,
        "frames_with_face": face_frames,
        "face_frame_ratio": round(face_frame_ratio, 3),
        "presence_threshold_seconds": min_presence_seconds,
        "note": "Anonymous face detections only; no identity recognition."
    }
    with open(OUTPUT_DIR / "attendance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=attendance_row.keys())
        writer.writeheader()
        writer.writerow(attendance_row)
    return attendance_row


def analyze_wav(path, window_seconds=1.0):
    """Compute RMS energy in windows for a WAV file. Energy is not speech recognition."""
    with wave.open(str(path), "rb") as wav:
        channels, sample_width, sample_rate, frames = (
            wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        )
        raw = wav.readframes(frames)
    if sample_width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    else:
        raise ValueError("Supported WAV sample widths: 8-bit, 16-bit, or 32-bit PCM.")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples /= max(float(np.max(np.abs(samples))) if samples.size else 1.0, 1.0)
    chunk_size = max(int(sample_rate * window_seconds), 1)
    rows = []
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start:start + chunk_size]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        rows.append({
            "start_seconds": round(start / sample_rate, 2),
            "end_seconds": round(min(start + chunk_size, len(samples)) / sample_rate, 2),
            "rms_energy": round(rms, 5),
            "activity_detected": rms >= 0.025
        })
    out = OUTPUT_DIR / "audio_activity.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start_seconds", "end_seconds", "rms_energy", "activity_detected"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "audio_file": str(path),
        "sample_rate": sample_rate,
        "duration_seconds": round(len(samples) / sample_rate, 2),
        "active_windows": sum(r["activity_detected"] for r in rows),
        "total_windows": len(rows),
        "activity_ratio": round(sum(r["activity_detected"] for r in rows) / max(len(rows), 1), 3),
        "note": "RMS energy proxy only; cannot reliably distinguish speech from noise."
    }


def make_report(vision=None, audio=None):
    vision_ratio = float(vision.get("face_frame_ratio", 0)) if vision else 0.0
    audio_ratio = float(audio.get("activity_ratio", 0)) if audio else 0.0
    # If only one modality is present, score based only on that modality.
    vals = []
    if vision is not None:
        vals.append(vision_ratio)
    if audio is not None:
        vals.append(audio_ratio)
    score = round(100 * (sum(vals) / len(vals)), 1) if vals else 0.0
    suggestions = []
    if vision is not None and vision_ratio < 0.5:
        suggestions.append("Check camera framing and attendance visibility; consider a short participation check-in.")
    if audio is not None and audio_ratio < 0.2:
        suggestions.append("If appropriate, add a discussion prompt or pair activity; verify whether the audio contains speech before interpreting low activity.")
    if vision is not None and audio is not None and abs(vision_ratio - audio_ratio) > 0.4:
        suggestions.append("Visual and audio indicators differ substantially; review the recording and sensor conditions before drawing conclusions.")
    if not suggestions:
        suggestions.append("Indicators do not show a strong alert. Use a quick formative question to confirm learner understanding.")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": "Multimodal Attendance & Engagement Agent",
        "prototype_engagement_indicator_percent": score,
        "vision": vision,
        "audio": audio,
        "teaching_suggestions": suggestions,
        "limitations": [
            "This score is a demonstration indicator, not a validated measure of attention.",
            "Face detections do not identify people or prove individual attendance.",
            "Audio energy does not reliably distinguish speech from noise.",
            "Use only consented media and follow institutional privacy requirements."
        ]
    }
    with open(OUTPUT_DIR / "session_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def demo_mode():
    # Demonstration data is explicitly simulated and is not real student data.
    vision = {"source": "SIMULATED DEMO DATA", "duration_seconds": 600,
              "frames_processed": 300, "frames_with_face": 246,
              "face_frame_ratio": 0.82,
              "note": "Simulated example only; not an actual attendance record."}
    audio = {"audio_file": "SIMULATED DEMO DATA", "duration_seconds": 600,
             "active_windows": 390, "total_windows": 600, "activity_ratio": 0.65,
             "note": "Simulated activity only; not real speech analysis."}
    report = make_report(vision, audio)
    print(json.dumps(report, indent=2))
    print(f"\nDemo report saved to {OUTPUT_DIR / 'session_report.json'}")


def main():
    print("\nMULTIMODAL ATTENDANCE & ENGAGEMENT AGENT")
    print("1. Analyze webcam")
    print("2. Analyze a video file")
    print("3. Analyze a WAV audio file")
    print("4. Generate clearly labelled demo report")
    choice = input("Choose 1-4: ").strip()
    vision = audio = None
    try:
        if choice == "1":
            vision = detect_video(0)
        elif choice == "2":
            path = input("Enter video file path: ").strip().strip('"')
            vision = detect_video(path)
        elif choice == "3":
            path = input("Enter WAV file path: ").strip().strip('"')
            audio = analyze_wav(path)
        elif choice == "4":
            demo_mode()
            return
        else:
            print("Invalid choice.")
            return
        report = make_report(vision, audio)
        print(json.dumps(report, indent=2))
        print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    except Exception as exc:
        print(f"Could not complete analysis: {exc}")


if __name__ == "__main__":
    main()
