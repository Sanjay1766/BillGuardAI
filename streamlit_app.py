
import streamlit as st
import os, json
from datetime import datetime
from PIL import Image
from ai_service import analyze_billboard
import pandas as pd
import plotly.express as px


if "history" not in st.session_state:
    st.session_state.history = []
if "credits" not in st.session_state:
    st.session_state.credits = 0
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False
if "accepted_disclaimer" not in st.session_state:
    st.session_state.accepted_disclaimer = False

st.set_page_config(page_title="Billboard Compliance Analyzer", layout="wide")


st.markdown("""
<style>
  .main { background: #f8fafc; }
  h2, h3 { margin-top: 0.2rem; }

  /* Two-panel main area */
  .grid-2 {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
  }
  .card {
    background: #fff; border-radius: 10px;
    padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }

  .inputs-row .stSelectbox, .inputs-row .stTextInput, .inputs-row .stFileUploader {
    margin-bottom: 0.25rem !important;
  }

  /* Buttons */
  .bottom-bar { display:flex; gap:10px; justify-content:center; margin-top: 6px; flex-wrap: wrap; }
  .stButton>button {
    background:#2563eb; color:#fff; border-radius:8px;
    padding:0.55em 1.1em; font-weight:600; font-size:15px;
  }
  .stButton>button:hover { background:#1e40af; }

  /* Result table */
  .analysis-table { font-size: 14px; border-collapse: collapse; width: 100%; }
  .analysis-table th, .analysis-table td { border: 1px solid #e5e7eb; padding: 6px; text-align: center; }
  .illegal { background: #fee2e2; color:#991b1b; font-weight:700; }
  .legal { background: #dcfce7; color:#166534; font-weight:700; }
  .unknown { background: #f3f4f6; color:#374151; }

  /* Dashboard cards grid */
  .dash-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px; }
  .stat-card {
    background:#fff; border-radius:10px; padding:14px; text-align:center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .stat-title { font-size:12px; color:#6b7280; margin-bottom:4px; }
  .stat-value { font-size:22px; font-weight:700; }

  .charts-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }

  /* Mobile stacking */
  @media (max-width: 900px) {
    .grid-2 { grid-template-columns: 1fr; }
    .dash-grid { grid-template-columns: 1fr 1fr; }
    .charts-grid { grid-template-columns: 1fr; }
  }
  @media (max-width: 600px) {
    .dash-grid { grid-template-columns: 1fr; }
  }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>📢 Billboard Compliance Analyzer</h2>", unsafe_allow_html=True)


if not st.session_state.accepted_disclaimer:
    st.warning("**⚠️ Privacy Disclaimer**\n\nBy uploading images, you agree that they are processed only for analysis and are **not stored permanently**.")
    if st.button("I Agree"):
        st.session_state.accepted_disclaimer = True
        st.rerun()
    st.stop()


st.markdown("<div class='inputs-row'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([2,2,1.5,1.5,2])
with c1: email = st.text_input("Email")
with c2: location = st.text_input("Location")
with c3: city = st.selectbox("City", ["Chennai","Delhi","Mumbai","Pune","Kolkata","Bengaluru","Hyderabad","Ahmedabad"])
with c4: area_type = st.selectbox("Area", ["Urban","Rural","Highway"])
with c5: uploaded_file = st.file_uploader("Select Image", type=["png","jpg","jpeg"])
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='grid-2'>", unsafe_allow_html=True)


st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Original Image")
if uploaded_file:
    st.image(uploaded_file, use_column_width=True)
else:
    st.info("Upload an image to analyze.")
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Analysis Results")
analyze_clicked = st.button("🚀 Analyze Billboard", use_container_width=True)

if uploaded_file and analyze_clicked:
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
        result = analyze_billboard(pil_img, state=city, area_type=area_type, visualize=True)

        analysis = result.get("analysis", {})
        if "error" in analysis:
            st.error(f"❌ Analysis failed: {analysis['error']}")
            st.stop()

        legal_status = str(analysis.get("legal_status", "unknown")).lower()
        reason = analysis.get("reason", "No reason provided")
        width = round(analysis.get("billboard_width_m", 0.0), 2)
        height = round(analysis.get("billboard_height_m", 0.0), 2)
        area_pct = round(result.get("billboard_area_percentage", 0.0), 1)
        angle = round(analysis.get("tilt_angle", 0.0), 1)
        illegal_text = analysis.get("illegal_text", "None")

        
        if legal_status == "illegal":
            st.error(f"🚨 NON-COMPLIANT — {reason}")
            if illegal_text and illegal_text != "None":
                st.markdown(
                    f"<p style='color:#b91c1c;font-weight:700;margin-top:-4px;'>🚫 Illegal Text: {illegal_text}</p>",
                    unsafe_allow_html=True
                )
            st.session_state.credits += 10
            delta = "+10"
        elif legal_status == "legal":
            st.success(f"✅ COMPLIANT — {reason}")
            delta = "+0"
        else:
            st.warning(f"⚠️ UNKNOWN STATUS — {reason}")
            delta = "+0"

        st.metric("⭐ Total Credits", st.session_state.credits, delta)

        
        df = pd.DataFrame([[
            legal_status, reason, width, height, area_pct, angle
        ]], columns=["Status","Reason","Width (m)","Height (m)","Area %","Tilt Angle"])
        row_class = "illegal" if legal_status=="illegal" else ("legal" if legal_status=="legal" else "unknown")
        st.markdown(df.to_html(classes=f"analysis-table {row_class}", index=False, escape=False), unsafe_allow_html=True)

      
        st.session_state.history.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "City": city, "Area": area_type,
            "Status": "VIOLATION" if legal_status=="illegal" else ("COMPLIANT" if legal_status=="legal" else "UNKNOWN"),
            "Reason": reason,
            "Width (m)": width, "Height (m)": height,
            "Area %": area_pct,
            "Tilt Angle": angle,
            "Reporter": email, "Location": location,
            "Illegal Text": illegal_text if legal_status == "illegal" else "",
            "Credits": 10 if legal_status == "illegal" else 0
        })
    except Exception as e:
        st.error(f"❌ Error during analysis: {e}")
elif not uploaded_file:
    st.info("Select an image and click **Analyze Billboard**.")

st.markdown("</div>", unsafe_allow_html=True)  # end right card
st.markdown("</div>", unsafe_allow_html=True)  # end grid-2


st.markdown("<div class='bottom-bar'>", unsafe_allow_html=True)
b1, b2, b3 = st.columns([1,1,1])

with b1:
    if st.session_state.history:
        report_json = json.dumps(st.session_state.history, indent=2)
        st.download_button("📑 Export Report", report_json, file_name="billboard_report.json")

with b2:
    if st.button("📊 View Dashboard"):
        st.session_state.show_dashboard = True

with b3:
    st.metric("⭐ Credits", st.session_state.credits)
st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.show_dashboard:
    st.markdown("### 📊 Reports Dashboard")

    if not st.session_state.history:
        st.info("No reports yet. Upload and analyze an image to see the dashboard.")
    else:
        df = pd.DataFrame(st.session_state.history)

        
        total_reports = len(df)
        violations = (df["Status"] == "VIOLATION").sum()
        compliant = (df["Status"] == "COMPLIANT").sum()
        total_credits = int(df["Credits"].sum())

        st.markdown("<div class='dash-grid'>", unsafe_allow_html=True)
        for title, value in [
            ("Total Reports", total_reports),
            ("Violations", violations),
            ("Compliant", compliant),
            ("Total Credits", total_credits),
        ]:
            st.markdown(
                f"<div class='stat-card'><div class='stat-title'>{title}</div><div class='stat-value'>{value}</div></div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        
        if total_reports > 0:
            st.markdown("<div class='charts-grid'>", unsafe_allow_html=True)
            colA, colB = st.columns(2)

            with colA:
                comp_counts = df["Status"].value_counts().reset_index()
                comp_counts.columns = ["Status", "Count"]
                fig = px.pie(comp_counts, names="Status", values="Count", title="Compliance Breakdown")
                st.plotly_chart(fig, use_container_width=True)

            with colB:
                credits_line = df[["Date","Credits"]].copy()
                credits_line["Date"] = pd.to_datetime(credits_line["Date"], format="%Y-%m-%d %H:%M")
                credits_line = credits_line.sort_values("Date")
                credits_line["Cumulative Credits"] = credits_line["Credits"].cumsum()
                fig2 = px.line(credits_line, x="Date", y="Cumulative Credits", title="Credits Over Time", markers=True)
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

       
        with st.expander("📄 View Report History", expanded=False):
            st.dataframe(df, use_container_width=True)

        if st.button("Hide Dashboard"):
            st.session_state.show_dashboard = False
