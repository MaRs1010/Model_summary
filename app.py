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

    prompt = f"""
Analyze the following content and generate EXACTLY 4 insights.

Format STRICTLY like this:

Insight 1: <Short Title>
- <Detailed explanation>

Insight 2: <Short Title>
- <Detailed explanation>

Insight 3: <Short Title>
- <Detailed explanation>

Insight 4: <Short Title>
- <Detailed explanation>

Rules:
- No repetition
- Each insight must be meaningful
- Use professional tone
- Avoid generic sentences

Content:
{text}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        num_beams=4
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return convert_to_bullets(result)

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

        if st.button("Extract & Summarize 🧠"):
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_image(image)

            st.subheader("📄 Extracted Text")
            st.write(extracted_text)

            with st.spinner("Summarizing..."):
                result = summarize_text(extracted_text)

            st.subheader("📌 Summary")
            for line in result.split("\n"):
                st.write(line)
