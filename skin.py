import streamlit as st
import base64
import os
import tempfile
from openai import OpenAI

# App title
st.set_page_config(page_title="AI Doctor - Medical Image Analysis", layout="centered")
st.title("🩺 AI Doctor - Medical Image Analysis")

# Sidebar for API key
st.sidebar.title("🔐 API Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Validate API key
if api_key and api_key.startswith("sk-"):
    client = OpenAI(api_key=api_key)
else:
    st.sidebar.warning("Please enter a valid OpenAI API key starting with 'sk-'.")
    st.stop()

# Instructions
with st.expander("ℹ️ About this App"):
    st.write(
        "Upload a medical image to get a detailed AI-generated analysis using GPT-4o. "
        "You can also get a simplified version using ELI5 (Explain Like I'm 5)."
    )

# Sample medical prompt
sample_prompt = """You are a medical practitioner and an expert in analyzing medical related images working for a very reputed hospital. 
You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result 
in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human 
body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

Now analyze the image and answer the above questions in the same structured manner defined above."""

# Session state init
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "result" not in st.session_state:
    st.session_state.result = None


# Utility functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_gpt4_model_for_analysis(filename: str, prompt=sample_prompt):
    base64_image = encode_image(filename)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, max_tokens=1500
    )
    return response.choices[0].message.content


def chat_eli(query):
    eli5_prompt = "You have to explain the below piece of information to a five years old.\n" + query
    messages = [{"role": "user", "content": eli5_prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=1500
    )
    return response.choices[0].message.content


# File uploader
uploaded_file = st.file_uploader("📤 Upload a Medical Image", type=["jpg", "jpeg", "png", "jfif"])

# Temporary file handling
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state["filename"] = tmp_file.name

    st.image(uploaded_file, caption="📸 Uploaded Image")

# Analyze button
if st.button("🧠 Analyze Image"):
    if "filename" in st.session_state and os.path.exists(st.session_state["filename"]):
        with st.spinner("Analyzing the image with GPT-4o..."):
            st.session_state["result"] = call_gpt4_model_for_analysis(st.session_state["filename"])
            os.unlink(st.session_state["filename"])  # Clean up temp file

# Display result
if st.session_state["result"]:
    st.subheader("📋 AI Medical Report")
    st.markdown(st.session_state["result"], unsafe_allow_html=True)

    st.info("Need a simpler explanation?")
    if st.radio("🧒 ELI5 - Explain Like I'm 5", ("No", "Yes")) == "Yes":
        with st.spinner("Simplifying the report..."):
            simplified = chat_eli(st.session_state["result"])
        st.subheader("🍼 ELI5 Explanation")
        st.markdown(simplified, unsafe_allow_html=True)
