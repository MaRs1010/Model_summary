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
# Step 1: Extract Data
# -----------------------------
def extract_data(text):
    numbers = re.findall(r'\$?\d+(?:\.\d+)?%?', text)

    keywords = {
        "revenue": bool(re.search(r"revenue|sales", text, re.I)),
        "profit": bool(re.search(r"profit|income", text, re.I)),
        "expense": bool(re.search(r"expense|cost", text, re.I))
    }

    return {
        "numbers": numbers,
        "keywords": keywords
    }

# -----------------------------
# Step 2: Build Prompt
# -----------------------------
def build_prompt(text, extracted):

    return f"""
You are a business analyst.

Generate EXACTLY 4 high-quality insights.

Each insight must:
- Be analytical (not descriptive)
- Use numbers if available: {extracted['numbers']}
- Avoid repetition
- Sound professional

Format EXACTLY like:

### 💡 Insight 1: <Short Title>
- <Explanation>

### 📊 Insight 2: <Short Title>
- <Explanation>

### 💸 Insight 3: <Short Title>
- <Explanation>

### 🔍 Insight 4: <Short Title>
- <Explanation>

Data:
{text}
"""

# -----------------------------
# Step 3: Generate Insights
# -----------------------------
def generate_insights(prompt):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Step 4: Clean Output
# -----------------------------
def clean_output(text):

    lines = text.split("\n")

    cleaned = []
    seen = set()

    for line in lines:
        line = line.strip()

        if len(line) > 10 and line not in seen:
            cleaned.append(line)
            seen.add(line)

    return "\n".join(cleaned)

# -----------------------------
# Main Pipeline
# -----------------------------
def get_insights(text):

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
st.markdown("Generate **4 powerful insights** from Text or Image")

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
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_image(image)

            st.subheader("📄 Extracted Text")
            st.write(extracted_text)

            with st.spinner("Generating insights..."):
                result = get_insights(extracted_text)

            st.markdown("## 📌 Insights")
            st.markdown(result)
