import streamlit as st
import base64
import os
import tempfile
from openai import OpenAI

# Set Streamlit page config
st.set_page_config(page_title="AI Doctor - Medical Image Analysis", layout="centered")

# --- Sidebar for API Key Input ---
st.sidebar.title("🔐 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Initialize OpenAI client only if API key is provided
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()
else:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# --- Session State Initialization ---
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

# --- Prompt for Medical Image Analysis ---
sample_prompt = """
You are a medical practitioner and an expert in analyzing medical-related images working for a reputed hospital.
You will be provided with images and need to identify anomalies, diseases, or health issues. Generate a detailed report
including all findings, next steps, and recommendations.

Important Notes:
- Only respond if the image is related to the human body and health issues.
- If something is unclear, say: 'Unable to determine based on the provided image.'
- Always include this disclaimer at the end: "Consult with a Doctor before making any decisions."
"""

# --- Encode Image to Base64 ---
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- GPT-4o Medical Image Analysis ---
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
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1500
    )

    return response.choices[0].message.content

# --- GPT-3.5 Simplified Explanation ---
def chat_eli(query):
    eli5_prompt = "You have to explain the below piece of information to a five years old:\n\n" + query
    messages = [{"role": "user", "content": eli5_prompt}]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1500
    )

    return response.choices[0].message.content

# --- Streamlit UI ---
st.title("🧠 AI Doctor - Medical Image Analysis")

with st.expander("ℹ️ About this App"):
    st.write("""
        Upload a medical image, and GPT-4o will analyze it for any visible health conditions or anomalies.
        You can also get a simplified ELI5 (Explain Like I'm 5) version of the diagnosis.
    """)

uploaded_file = st.file_uploader("📤 Upload a Medical Image", type=["jpg", "jpeg", "png", "jfif"])

# --- Temporary File Handling ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state['filename'] = tmp_file.name

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

# --- Analyze Image Button ---
if st.button('🔍 Analyze Image'):
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        st.session_state['result'] = call_gpt4_model_for_analysis(st.session_state['filename'])
        st.markdown(st.session_state['result'], unsafe_allow_html=True)
        os.unlink(st.session_state['filename'])  # Clean up

# --- ELI5 Explanation ---
if 'result' in st.session_state and st.session_state['result']:
    st.info("Would you like a simplified explanation?")
    if st.radio("ELI5 - Explain Like I'm 5", ('No', 'Yes')) == 'Yes':
        simplified_explanation = chat_eli(st.session_state['result'])
        st.markdown(simplified_explanation, unsafe_allow_html=True)
