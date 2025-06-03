import streamlit as st
import base64
import os
from openai import OpenAI
import tempfile

# App Configuration
st.set_page_config(
    page_title="AI Doctor - Medical Image Analysis",
    page_icon="🏥",
    layout="wide"
)

sample_prompt = """You are a medical practictioner and an expert in analzying medical related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

Now analyze the image and answer the above questions in the same structured manner defined above."""

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

def get_openai_client(api_key):
    """Initialize OpenAI client with provided API key"""
    try:
        # Try to get API key from Streamlit secrets first, then from user input
        if not api_key:
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
            except:
                return None
        
        if api_key and api_key.strip():
            return OpenAI(api_key=api_key.strip())
        return None
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return None

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4_model_for_analysis(filename: str, client, sample_prompt=sample_prompt):
    """Analyze image using GPT-4 Vision"""
    try:
        if not client:
            st.error("OpenAI client not initialized. Please check your API key.")
            return None
            
        base64_image = encode_image(filename)
        
        messages = [
            {
                "role": "user",
                "content":[
                    {
                        "type": "text", 
                        "text": sample_prompt
                    },
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
        
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

def chat_eli(query, client):
    """Generate ELI5 explanation"""
    try:
        if not client:
            st.error("OpenAI client not initialized. Please check your API key.")
            return None
            
        eli5_prompt = "You have to explain the below piece of information to a five years old. \n" + query
        messages = [
            {
                "role": "user",
                "content": eli5_prompt
            }
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500
        )

        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating ELI5 explanation: {str(e)}")
        return None

def validate_api_key(api_key):
    """Validate OpenAI API key by making a simple request"""
    try:
        client = OpenAI(api_key=api_key)
        # Make a simple request to validate the key
        response = client.models.list()
        return True
    except Exception as e:
        return False

# Sidebar for API Key and Instructions
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # API Key Input
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        value=st.session_state.api_key,
        help="Your API key is not stored and only used for this session",
        placeholder="sk-..."
    )
    
    # Update session state when API key changes
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
    
    # Validate API key button
    if st.button("🔍 Validate API Key", type="secondary"):
        if api_key_input:
            with st.spinner("Validating API key..."):
                if validate_api_key(api_key_input):
                    st.success("✅ API key is valid!")
                else:
                    st.error("❌ Invalid API key. Please check and try again.")
        else:
            st.warning("Please enter an API key first.")
    
    # API Key Status
    try:
        # Check if we have a valid API key from secrets or user input
        has_secrets_key = False
        try:
            secrets_key = st.secrets["OPENAI_API_KEY"]
            if secrets_key:
                has_secrets_key = True
        except:
            pass
            
        if has_secrets_key:
            st.info("🔐 Using configured API key from app secrets")
        elif api_key_input:
            st.info("🔑 Using your provided API key")
        else:
            st.warning("⚠️ No API key provided")
            
    except:
        if api_key_input:
            st.info("🔑 Using your provided API key")
        else:
            st.warning("⚠️ No API key provided")
    
    st.divider()
    
    # Instructions
    st.header("📋 How to Get API Key")
    st.markdown("""
    1. Go to [OpenAI Platform](https://platform.openai.com/)
    2. Sign up or log in to your account
    3. Navigate to **API Keys** section
    4. Click **"Create new secret key"**
    5. Copy the key and paste it above
    
    **💡 Note:** Your API key is only used during your current session and is not stored anywhere.
    """)
    
    st.divider()
    
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Enter** your OpenAI API key above
    2. **Upload** a medical image
    3. **Click** 'Analyze Image' button
    4. **Review** the detailed analysis
    5. **Optional:** Get a simplified explanation
    """)
    
    st.divider()
    
    st.header("⚠️ Important Disclaimer")
    st.markdown("""
    This AI tool is for **educational and informational purposes only**. 
    
    It should **NOT replace professional medical advice**, diagnosis, or treatment. 
    
    **Always consult with qualified healthcare professionals** for medical concerns.
    """)

# Main App Interface
st.title("🏥 AI Doctor - Medical Image Analysis")

# Check if we have a valid API key
current_api_key = ""
try:
    # Try to get from secrets first
    current_api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Fall back to user input
    current_api_key = st.session_state.api_key

# Initialize OpenAI client
client = get_openai_client(current_api_key)

if not client:
    st.error("🚫 **No valid API key found!** Please enter your OpenAI API key in the sidebar to use this application.")
    st.info("👈 Use the sidebar to enter your OpenAI API key")
    st.stop()

with st.expander("ℹ️ About this App"):
    st.write("""
    This application uses **GPT-4 Vision** to analyze medical images and provide detailed insights.
    
    **Features:**
    - 🔍 Upload medical images for AI analysis
    - 📋 Get detailed findings and recommendations  
    - 🧒 ELI5 (Explain Like I'm 5) option for simplified explanations
    - 🔐 Secure API key handling
    
    **Supported Image Formats:** JPG, JPEG, PNG, JFIF
    
    **⚠️ Disclaimer:** This tool is for educational purposes only. Always consult with a qualified healthcare professional for medical advice.
    """)

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload a Medical Image", 
    type=["jpg", "jpeg", "png", "jfif"],
    help="Supported formats: JPG, JPEG, PNG, JFIF (Max size: 200MB)"
)

# Display uploaded image
if uploaded_file is not None:
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state['filename'] = tmp_file.name

        # Display image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption='📷 Uploaded Medical Image', use_column_width=True)
            
        # Show file info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
        st.info(f"📊 **File Info:** {uploaded_file.name} ({file_size:.2f} MB)")
            
    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {str(e)}")

# Analysis section
if uploaded_file is not None:
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button('🔍 Analyze Image', type="primary", use_container_width=True)

    if analyze_button:
        if not client:
            st.error("🚫 Cannot analyze image: No valid API key provided.")
        elif 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
            with st.spinner('🔄 Analyzing image... This may take a few moments.'):
                result = call_gpt4_model_for_analysis(st.session_state['filename'], client)
                
                if result:
                    st.session_state['result'] = result
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display results
                    st.subheader("📋 Medical Analysis Results")
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                            {result}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Clean up temporary file
                    try:
                        os.unlink(st.session_state['filename'])
                    except:
                        pass
                else:
                    st.error("❌ Failed to analyze the image. Please check your API key and try again.")
        else:
            st.error("❌ No valid image file found. Please upload an image first.")

# ELI5 Explanation section
if 'result' in st.session_state and st.session_state['result'] and client:
    st.divider()
    st.subheader("🧒 Simplified Explanation (ELI5)")
    st.info("💡 Get a simplified explanation that's easy to understand!")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        eli5_button = st.button("🌟 Generate ELI5 Explanation", type="secondary", use_container_width=True)
    
    if eli5_button:
        with st.spinner('🔄 Generating simplified explanation...'):
            simplified_explanation = chat_eli(st.session_state['result'], client)
            
            if simplified_explanation:
                st.markdown("### 🌟 Easy to Understand Version:")
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                        {simplified_explanation}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            else:
                st.error("❌ Failed to generate simplified explanation. Please try again.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🏥 <strong>AI Doctor - Medical Image Analysis</strong></p>
    <p>⚠️ <strong>Remember:</strong> This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
    <p>🔐 Your API key is secure and only used during your current session.</p>
</div>
""", unsafe_allow_html=True)
