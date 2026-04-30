import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import pytesseract
import re

# Fix for Streamlit Cloud OCR
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# -----------------------------
# Load Model (FLAN-T5)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# OCR
# -----------------------------
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# -----------------------------
# Clean OCR Text
# -----------------------------
def clean_ocr_text(text):
    text = re.sub(r'[^A-Za-z0-9$%., ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# Extract Data
# -----------------------------
def extract_data(text):
    numbers = re.findall(r'\$?\d+(?:\.\d+)?%?', text)

    return {
        "numbers": numbers
    }

# -----------------------------
# Build Prompt
# -----------------------------
def build_prompt(text, extracted):

    return f"""
You are a financial analyst.

Generate EXACTLY 4 meaningful business insights.

Each insight should:
- Have a short title
- Provide a clear explanation
- Use numbers if available: {extracted['numbers']}
- Avoid repetition

Format:

Insight 1: Title
- Explanation

Insight 2: Title
- Explanation

Insight 3: Title
- Explanation

Insight 4: Title
- Explanation

Data:
{text}
"""

# -----------------------------
# Generate Insights
# -----------------------------
def generate_insights(prompt):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Clean Output
# -----------------------------
def clean_output(text):

    insights = re.findall(r'Insight \d+:.*?(?=Insight \d+:|$)', text, re.S)

    cleaned = []
    seen = set()

    for ins in insights:
        ins = ins.strip()

        if ins and ins not in seen:
            cleaned.append(ins)
            seen.add(ins)

    return "\n\n".join(cleaned[:4])

# -----------------------------
# Main Pipeline
# -----------------------------
def get_insights(text):

    text = clean_ocr_text(text)

    extracted = extract_data(text)

    prompt = build_prompt(text, extracted)

    raw_output = generate_insights(prompt)

    final_output = clean_output(raw_output)

    return final_output

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Insight Generator", layout="wide")

st.title("🧠 AI Business Insight Generator")
st.markdown("Generate **4 smart insights** from text or images")

tab1, tab2 = st.tabs(["📝 Text Input", "🖼️ Image Upload"])

# -----------------------------
# TEXT TAB
# -----------------------------
with tab1:
    user_text = st.text_area("Enter your data:", height=250)

    if st.button("Generate Insights 🚀"):
        if user_text.strip() == "":
            st.warning("Please enter text")
        else:
            with st.spinner("Analyzing..."):
                result = get_insights(user_text)

            st.markdown("## 📌 Insights")
            st.markdown(result)

# -----------------------------
# IMAGE TAB
# -----------------------------
with tab2:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        if st.button("Extract & Analyze 🧠"):
            with st.spinner("Processing..."):
                extracted_text = extract_text_from_image(image)
                result = get_insights(extracted_text)

            st.markdown("## 📌 Insights")
            st.markdown(result)
