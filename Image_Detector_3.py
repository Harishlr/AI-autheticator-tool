import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
import tempfile
import os

# 1. Page Setup
st.set_page_config(page_title="ACEPRO Authenticator", page_icon="🛡️", layout="centered")

# Mobile-First Styling
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #004a99; color: white; }
    .stHeader { font-size: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model_a = pipeline("image-classification", model="organika/sdxl-detector")
    model_b = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    return model_a, model_b

m_a, m_b = load_models()

st.title("🛡️ ACEPRO Mobile Authenticator")

# --- AUTO-RUN LOGIC ---
# The 'label_visibility' makes it cleaner on small screens
img_file = st.file_uploader("Select or Capture Media", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

if img_file:
    img = Image.open(img_file)
    st.image(img, use_container_width=True)
    
    # We remove the "Run Analysis" button and go straight to work
    with st.spinner("🔍 Analyzing origin..."):
        res_a = m_a(img)[0]
        res_b = m_b(img)[0]
        
        is_ai_a = res_a['label'].lower() in ['artificial', 'generated', 'fake']
        is_ai_b = res_b['label'].lower() in ['artificial', 'generated', 'fake']

        st.divider()
        
        if is_ai_a and is_ai_b:
            st.error(f"### Verdict: HIGH PROBABILITY AI ({res_a['score']:.1%})")
        elif is_ai_a or is_ai_b:
            st.warning("### Verdict: INCONCLUSIVE / EDITED")
            st.write("Models show conflicting patterns. Likely a real photo with AI filters.")
        else:
            st.success(f"### Verdict: LIKELY REAL ({1 - res_a['score']:.1%})")

        # Quick Forensic Breakdown
        with st.expander("See Technical Details"):
            st.write(f"Primary Scan: {res_a['label']} at {res_a['score']:.2%}")
            st.write(f"Secondary Scan: {res_b['label']} at {res_b['score']:.2%}")
            
        # Download button remains for professional records
        report = f"ACEPRO Report\nFile: {img_file.name}\nResult: {res_a['label']}"
        st.download_button("📩 Save Analysis", report, file_name="forensic_result.txt")

else:
    st.info("💡 Tip: On mobile, you can snap a photo of a computer screen or TV to check for AI artifacts.")
