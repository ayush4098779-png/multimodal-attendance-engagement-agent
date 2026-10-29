import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path("outputs")
st.set_page_config(page_title="Multimodal Attendance Agent", page_icon="📊", layout="wide")
st.title("Multimodal Attendance & Engagement Agent")
st.caption("Local prototype • transparent indicators • not a validated attention detector")

st.info("Privacy note: do not upload identifiable student media without permission. This dashboard reads local output files only.")
report_path = OUT / "session_report.json"
attendance_path = OUT / "attendance.csv"
audio_path = OUT / "audio_activity.csv"

if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    st.metric("Prototype engagement indicator", f"{report.get('prototype_engagement_indicator_percent', 0)}%")
    left, right = st.columns(2)
    with left:
        st.subheader("Vision summary")
        if report.get("vision"):
            st.json(report["vision"])
        else:
            st.write("No vision analysis in this report.")
    with right:
        st.subheader("Audio summary")
        if report.get("audio"):
            st.json(report["audio"])
        else:
            st.write("No audio analysis in this report.")
    st.subheader("Teaching suggestions")
    for item in report.get("teaching_suggestions", []):
        st.write(f"- {item}")
    with st.expander("Limitations and responsible use"):
        for item in report.get("limitations", []):
            st.write(f"- {item}")
else:
    st.warning("No session report found. Run `python main.py` first and choose demo mode or analyze media.")
    if st.button("Load clearly labelled demo report"):
        from main import demo_mode
        demo_mode()
        st.rerun()

if attendance_path.exists():
    st.subheader("Attendance/session output")
    st.dataframe(pd.read_csv(attendance_path), use_container_width=True)
if audio_path.exists():
    st.subheader("Audio activity windows")
    audio_df = pd.read_csv(audio_path)
    st.dataframe(audio_df, use_container_width=True)
    st.line_chart(audio_df.set_index("start_seconds")[["rms_energy"]])
