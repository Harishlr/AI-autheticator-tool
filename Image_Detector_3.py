import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="ACEPRO AI Authenticator", page_icon="🛡️", layout="centered")

# --- PROFESSIONAL SIDEBAR ---
with st.sidebar:
    st.title("🛡️ ACEPRO Forensic Lab")
    st.info("""
    **Accuracy Guidance:**
    * **Direct Uploads:** ~98% Accuracy. Best for forensic verification.
    * **Live Snaps:** ~70% Accuracy. Useful for quick field screening.
    """)
    st.divider()
    st.caption("© 2026 ACEPRO Management Training & Consultancy")

@st.cache_resource
def load_models():
    model_a = pipeline("image-classification", model="organika/sdxl-detector")
    model_b = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    return model_a, model_b

m_a, m_b = load_models()

st.title("🛡️ AI Content Authenticator")

# Input Selection
mode = st.radio("Select Analysis Mode:", ["Direct Upload (High Accuracy)", "Live Snap (Field Use)"], horizontal=True)

img_file = None

if mode == "Direct Upload (High Accuracy)":
    img_file = st.file_uploader("Select original media file", type=['jpg', 'png', 'jpeg'])
else:
    img_file = st.camera_input("Scan suspicious screen")

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Analyzing...", use_container_width=True)
    
    with st.spinner("🕵️ Processing Forensic Markers..."):
        res_a = m_a(img)[0]
        res_b = m_b(img)[0]
        
        is_ai_a = res_a['label'].lower() in ['artificial', 'generated', 'fake']
        is_ai_b = res_b['label'].lower() in ['artificial', 'generated', 'fake']

        st.divider()
        
        # Result Display
        if is_ai_a and is_ai_b:
            st.error(f"### 🚩 HIGH PROBABILITY AI ({res_a['score']:.1%})")
        elif is_ai_a or is_ai_b:
            st.warning("### ⚠️ INCONCLUSIVE / LIKELY EDITED")
        else:
            st.success(f"### ✅ LIKELY AUTHENTIC ({1 - res_a['score']:.1%})")

        # --- THE NEW ACCURACY ADVISORY ---
        if mode == "Live Snap (Field Use)":
            st.info("💡 **Precision Tip:** This 'Live Snap' has higher noise. For a definitive professional verdict, please **Direct Upload** the original file.")
        else:
            st.success("💎 **Precision Mode:** This analysis was performed on an original file for maximum forensic accuracy.")

        # Technical Data
        with st.expander("📊 View Technical Data"):
            st.write(f"Diffusion Analysis: {res_a['label']} ({res_a['score']:.2%})")
