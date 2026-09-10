import streamlit as st
from PIL import Image
from src.preprocessor import preprocess_image
from src.classifier import classify_waste

st.set_page_config(page_title="AI Waste Segregator", page_icon="♻️", layout="wide")

st.title("♻️ AI Waste-Segregation & Recycling Assistant")
st.caption("Identify waste category, proper disposal rules, and upcycling ideas in seconds.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Input Waste Item")
    input_mode = st.radio("Choose Input:", ["Camera Capture", "Upload Image"], horizontal=True)
    uploaded_file = None

    if input_mode == "Camera Capture":
        uploaded_file = st.camera_input("Snap a photo of the item")
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Image", use_container_width=True)

with col_right:
    st.subheader("2. AI Analysis & Recycling Guide")
    if uploaded_file:
        if st.button("🔍 Analyze Waste", type="primary", use_container_width=True):
            with st.spinner("Processing image with OpenCV & AI..."):
                try:
                    # Preprocess with OpenCV
                    clean_img = preprocess_image(input_image)
                    
                    # Classify with AI
                    result = classify_waste(clean_img)

                    # Display Category Badge
                    color_badges = {
                        "Wet": "🟢 WET / ORGANIC WASTE",
                        "Recyclable": "🔵 RECYCLABLE WASTE",
                        "E-waste": "🟠 E-WASTE",
                        "Hazardous": "🔴 HAZARDOUS WASTE",
                        "General": "⚪ GENERAL / LANDFILL"
                    }
                    badge = color_badges.get(result.category, f"⚪ {result.category.upper()}")
                    st.success(f"### Category: {badge}")

                    # Item Details
                    st.markdown(f"**Identified Item:** {result.item_name} ({result.material})")
                    st.markdown(f"**Condition:** {result.condition}")
                    st.markdown(f"**Target Bin:** {result.disposal_bin}")

                    # Preparation Steps
                    st.markdown("#### 📋 Preparation & Disposal Instructions")
                    for step in result.preparation_steps:
                        st.write(f"- {step}")

                    # Upcycling & Eco Impact
                    st.markdown("#### 💡 Upcycling Idea")
                    st.info(result.upcycle_idea)

                    st.markdown("#### 🌱 Environmental Impact")
                    st.warning(result.environmental_fact)

                except Exception as e:
                    st.error(f"Error during analysis: {e}")
    else:
        st.info("Please upload an image or take a photo to begin.")