import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import pytesseract
import re

# Fix OCR path (for deployment environments)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# -----------------------------
# Load model (auto-downloads)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# OCR function
# -----------------------------
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# -----------------------------
# Convert summary → 4 bullets
# -----------------------------
def convert_to_bullets(text):
    sentences = re.split(r'(?<=[.!?]) +', text)

    bullets = []
    for s in sentences:
        s = s.strip()
        if len(s) > 20:
            bullets.append("• " + s)

    return "\n".join(bullets[:4])

# -----------------------------
# Summarization
# -----------------------------
def summarize_text(text):

    # Step 1: Clean summary (NO instructions inside)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        num_beams=4
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Step 2: Convert into structured insights
    import re
    sentences = re.split(r'(?<=[.!?]) +', summary)

    insights = []
    for i, s in enumerate(sentences):
        s = s.strip()
        if len(s) > 25:
            title = s.split(" ")[0:4]  # first few words as title
            title = " ".join(title)

            insights.append(f"{s}")

    return "\n\n".join(insights[:4])
# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Smart Summarizer", layout="wide")

st.title("🧠 AI Smart Summarizer")
st.markdown("Summarize **Text or Image → 4 Bullet Points**")

tab1, tab2 = st.tabs(["📝 Text", "🖼️ Image"])

# -----------------------------
# TEXT INPUT
# -----------------------------
with tab1:
    user_text = st.text_area("Enter your text:", height=250)

    if st.button("Generate Summary 🚀"):
        if user_text.strip() == "":
            st.warning("Please enter text")
        else:
            with st.spinner("Summarizing..."):
                result = summarize_text(user_text)

            st.subheader("📌 Summary")
            for line in result.split("\n"):
                st.write(line)

# -----------------------------
# IMAGE INPUT
# -----------------------------
with tab2:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        if st.button("Summarize 🧠"):
            with st.spinner("Extracting text..."): 
                extracted_text = extract_text_from_image(image) 
            st.subheader("📄 Extracted Text") 
            st.write(extracted_text)
            with st.spinner("Summarizing..."):
                result = summarize_text(extracted_text)

            st.subheader("📌 Summary")
            st.markdown(result)
