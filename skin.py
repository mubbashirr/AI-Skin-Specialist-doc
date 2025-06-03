import streamlit as st
import base64
import os
from openai import OpenAI
import tempfile

# --- App Title ---
st.set_page_config(page_title="AI Doctor - Medical Image Analysis", layout="wide")
st.title("🧠 AI Doctor - Medical Image Analysis")

# --- Sidebar for API Key ---
st.sidebar.header("🔐 Enter Your OpenAI API Key")
api_key = st.sidebar.text_input("API Key", type="password")

# Validate API key
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to use the app.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Sample Prompt ---
sample_prompt = """You are a medical practictioner and an expert in analzying medical related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

Now analyze the image and answer the above questions in the same structured manner defined above."""

# --- Session State ---
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

# --- Encode Image ---
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- GPT-4o Call with Vision ---
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

# --- ELI5 Explanation ---
def chat_eli(query):
    eli5_prompt = "You have to explain the below piece of information to a five years old.\n\n" + query
    messages = [{"role": "user", "content": eli5_prompt}]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1000
    )

    return response.choices[0].message.content

# --- About Section ---
with st.expander("ℹ️ About this App"):
    st.markdown("""
    Upload a medical image (X-ray, MRI, skin condition, etc.) and get a GPT-4-based analysis.
    \nNote: This is for **educational and research** purposes only. Always consult a certified physician for any diagnosis.
    """)

# --- Upload File ---
uploaded_file = st.file_uploader("📤 Upload a Medical Image", type=["jpg", "jpeg", "png", "jfif"])

# --- Save to Temp & Display ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state['filename'] = tmp_file.name
    st.image(uploaded_file, caption='🖼️ Uploaded Image Preview', use_column_width=True)

# --- Analyze Button ---
if st.button('🔍 Analyze Image'):
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        st.session_state['result'] = call_gpt4_model_for_analysis(st.session_state['filename'])
        st.markdown("### 🧾 Analysis Result:")
        st.markdown(st.session_state['result'], unsafe_allow_html=True)
        os.unlink(st.session_state['filename'])  # Clean up

# --- ELI5 Option ---
if st.session_state.get('result'):
    st.info("Want a simpler explanation?")
    if st.radio("📚 ELI5 - Explain Like I'm 5", ['No', 'Yes'], index=0) == 'Yes':
        explanation = chat_eli(st.session_state['result'])
        st.markdown("### 🍼 Simplified Explanation")
        st.markdown(explanation, unsafe_allow_html=True)
