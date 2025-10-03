import streamlit as st
import requests
from PIL import Image
import time
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="LCN Terminal",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for the Hacker Vibe ---
st.markdown("""
<style>
/* Main app background */
body, .main {
    background-color: #0d0208; /* Near-black background */
    color: #00ff41; /* Hacker green text */
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
}

/* Titles and Headers */
h1, h2, h3 {
    color: #00ff41; /* Hacker green */
    text-shadow: 0 0 5px #00ff41;
}

/* File Uploader */
.stFileUploader > label {
    font-size: 1.2rem;
    color: #00ff41;
}
.stFileUploader > div > div {
    border: 1px dashed #00ff41;
    background-color: #1a1a1a;
}

/* Buttons */
.stButton>button {
    border: 1px solid #00ff41;
    background-color: transparent;
    color: #00ff41;
    padding: 10px 20px;
    border-radius: 0; /* Sharp corners */
}
.stButton>button:hover {
    background-color: rgba(0, 255, 65, 0.2);
    color: #ffffff;
    border-color: #00ff41;
}

/* Expander for JSON */
.stExpander {
    border: 1px solid #00ff41;
    background-color: #1a1a1a;
    border-radius: 0;
}

/* Progress bar / spinner */
.stSpinner > div > div {
    border-top-color: #00ff41;
}

/* Success/Info/Error boxes */
.stAlert {
    border: 1px solid #00ff41;
    border-radius: 0;
}
</style>
""", unsafe_allow_html=True)


# --- API Configuration ---
API_URL = "http://127.0.0.1:8000/predict"

# --- UI Layout ---

# ASCII Art Header
st.code("""
  _      _____ _   _   _    _   _ _____ _______ 
 | |    / ____| \ | | | |  | \ | |_   _|__   __|
 | |   | |    |  \| | | |  |  \| | | |    | |   
 | |   | |    | . ` | | |  | . ` | | |    | |   
 | |___| |____| |\  | | |__| |\  |_| |_   | |   
 |______\_____|_| \_|  \____/_| \_|_____|  |_|   
                                              
 --- Land Cover Neural Network // GEOINT ANALYSIS TERMINAL ---
""", language='text')

st.header(">> Target Acquisition Module")

# File uploader widget
uploaded_file = st.file_uploader(
    "Drag and drop target image here or browse files...", 
    type=["jpg", "png", "tif", "tiff"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(Image.open(uploaded_file), caption="TARGET IMAGE MATRIX", use_column_width=True)

    with col2:
        # Simulate a fake "analysis" log for dramatic effect
        with st.spinner(''):
            st.info(">> EXECUTING DEEP SCAN...")
            time.sleep(1)
            st.info(">> INITIALIZING CONNECTION TO INFERENCE SERVER...")
            time.sleep(1)
            st.info(">> TRANSMITTING IMAGE MATRIX [ENCRYPTED]...")
            time.sleep(2)
            st.info(">> AWAITING NEURAL NETWORK RESPONSE...")

        # Send the file to the FastAPI backend
        files = {"file": uploaded_file.getvalue()}
        
        try:
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                class_name = result["predicted_class"]
                confidence = result["confidence"]
                
                st.success(">> ANALYSIS COMPLETE. SYSTEM OUTPUT:")
                
                # Display the result in a formatted code block
                output_text = f"""
                +----------------------------+
                | DECODED PREDICTION         |
                +----------------------------+
                | Target Class..: {class_name.upper()}
                | Confidence....: {confidence*100:.2f}%
                +----------------------------+
                """
                st.code(output_text, language="text")

                # Show the raw JSON in an expander
                with st.expander(">> VIEW RAW TRANSMISSION LOG"):
                    st.json(result)
                    
            else:
                st.error(f">> SERVER ERROR [CODE: {response.status_code}]")
                st.json(response.json())

        except requests.exceptions.ConnectionError:
            st.error(">> FATAL ERROR: CONNECTION TO INFERENCE SERVER FAILED. ENSURE BACKEND IS ONLINE.")