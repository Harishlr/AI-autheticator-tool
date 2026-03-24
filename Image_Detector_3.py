import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
import tempfile
import os

# 1. Page Setup & Professional Branding
st.set_page_config(page_title="ACEPRO AI Authenticator", page_icon="🛡️", layout="centered")

# Custom CSS for a professional "App" feel
st.markdown("""
    <style>
    .main { max-width: 600px; margin: 0 auto; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #004a99; color: white; font-weight: bold; }
    .stAlert { border-radius: 10px; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load the "Expert" Models (Cached for speed)
@st.cache_resource
def load_models():
    # Model A: Specialized in modern Diffusion (SDXL/Midjourney)
    model_a = pipeline("image-classification", model="organika/sdxl-detector")
    # Model B: General vision classifier for real vs fake
    model_b = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    return model_a, model_b

m_a, m_b = load_models()

# 3. Sidebar Navigation
st.sidebar.title("🛡️ Authentication Suite")
st.sidebar.info("Professional verification tool for media and text origin.")
mode = st.sidebar.radio("Select Tool:", ["AI Image/Video Detector", "Text Sentiment Pro"])

# --- TOOL 1: IMAGE & VIDEO ---
if mode == "AI Image/Video Detector":
    st.title("🔍 Origin Authenticator")
    st.write("Verify if media is **Camera-Captured** or **AI-Generated**.")
    
    tab1, tab2 = st.tabs(["🖼️ Image Analysis", "🎥 Video Analysis"])

    with tab1:
        img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if img_file:
            img = Image.open(img_file)
            st.image(img, use_container_width=True, caption="Uploaded Media")
            
            if st.button("Run Forensic Analysis"):
                with st.spinner("Analyzing pixel structures and frequency artifacts..."):
                    # Run both models for a "Second Opinion"
                    res_a = m_a(img)[0]
                    res_b = m_b(img)[0]
                    
                    # Logic: Create a consensus verdict
                    is_ai_a = res_a['label'].lower() in ['artificial', 'generated', 'fake']
                    is_ai_b = res_b['label'].lower() in ['artificial', 'generated', 'fake']

                    st.divider()
                    
                    if is_ai_a and is_ai_b:
                        st.error("### Verdict: HIGH PROBABILITY AI")
                        st.write("Both forensic models detected non-natural pixel patterns.")
                    elif is_ai_a or is_ai_b:
                        st.warning("### Verdict: INCONCLUSIVE / LIKELY EDITED")
                        st.write("Models disagree. This suggests a real photo with heavy AI filters or a very new AI model.")
                    else:
                        st.success("### Verdict: LIKELY REAL")
                        st.write("Patterns are consistent with standard camera sensor noise.")

                    # Metrics & Reporting
                    col1, col2 = st.columns(2)
                    col1.metric("Primary Confidence", f"{res_a['score']:.1%}")
                    col2.metric("Model Verdict", res_a['label'].title())

                    # Exportable Report
                    report = f"ACEPRO Forensic Report\nFile: {img_file.name}\nResult: {res_a['label']}\nConfidence: {res_a['score']:.2%}"
                    st.download_button("📩 Download Analysis Report", report, file_name="ai_forensic_report.txt")

    with tab2:
        st.info("Video analysis scans initial frames for synthetic artifacts.")
        vid_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])
        if vid_file and st.button("Scan Video Clip"):
            with st.spinner("Processing video frames..."):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(vid_file.read())
                cap = cv2.VideoCapture(tfile.name)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = m_a(Image.fromarray(frame_rgb))[0]
                    st.metric("Video Authenticity Score", f"{res['score']:.1%}", delta=res['label'])
                    st.video(vid_file)
                cap.release()
                os.unlink(tfile.name)

# --- TOOL 2: TEXT SENTIMENT ---
else:
    st.title("📊 Tone Intelligence")
    text_input = st.text_area("Enter correspondence for tone verification:", height=150, placeholder="Paste text here...")
    if st.button("Analyze Tone"):
        with st.spinner("Evaluating intent..."):
            # Placeholder for the sentiment logic we built earlier
            sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            result = sentiment_model(text_input)[0]
            st.write(f"### Detected Tone: {result['label'].title()}")
            st.progress(result['score'])