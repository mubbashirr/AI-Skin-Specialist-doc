import streamlit as st
import os
from openai import OpenAI
import base64
import tempfile

# Sidebar API key input
st.sidebar.title("API Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if not api_key or not api_key.startswith("sk-"):
    st.sidebar.warning("Please enter a valid API key to continue.")
    st.stop()

# Set the environment variable and initialize OpenAI client
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

# Session state init
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

sample_prompt = """You are a medical practitioner and an expert in analyzing medical-related images...
(keep rest of your original prompt here)
"""

# Functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4_model_for_analysis(filename, prompt=sample_prompt):
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

def chat_eli(query):
    eli5_prompt = "You have to explain the below piece of information to a five-year-old:\n" + query
    messages = [{"role": "user", "content": eli5_prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1500
    )
    return response.choices[0].message.content

# UI Layout
st.title("AI Doctor - Medical Image Analysis")

with st.expander("About this App"):
    st.write("Upload a medical image to receive a professional AI analysis using GPT-4o.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state['filename'] = tmp_file.name
    st.image(uploaded_file, caption='Uploaded Image')

if st.button("Analyze Image"):
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        st.session_state['result'] = call_gpt4_model_for_analysis(st.session_state['filename'])
        st.markdown(st.session_state['result'], unsafe_allow_html=True)
        os.unlink(st.session_state['filename'])

if st.session_state.get('result'):
    st.info("Would you like a simplified version of the result?")
    if st.radio("ELI5 - Explain Like I'm 5", ['No', 'Yes']) == 'Yes':
        simplified = chat_eli(st.session_state['result'])
        st.markdown(simplified, unsafe_allow_html=True)
