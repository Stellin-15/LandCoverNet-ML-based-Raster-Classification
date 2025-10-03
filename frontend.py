import streamlit as st
import requests
from PIL import Image

# --- Page Configuration ---
# Use a wide layout for a more modern feel
st.set_page_config(
    page_title="LandCoverNet",
    page_icon="🗺️",
    layout="centered",
)

# --- Custom CSS for the Google Maps Vibe ---
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #f5f5f5; /* Light grey background */
    }
    /* Main content card */
    .stApp > div:first-child > div:first-child > div:first-child {
        padding: 2rem;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Title and Header styles */
    h1 {
        color: #333333;
        font-family: 'sans-serif';
    }
    /* Button styles */
    .stButton>button {
        background-color: #1a73e8; /* Google's blue */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b66c9;
    }
    /* File uploader style */
    .stFileUploader label {
        font-size: 1.1rem;
        font-weight: bold;
        color: #5f6368;
    }
</style>
""", unsafe_allow_html=True)

# --- API Configuration ---
API_URL = "http://127.0.0.1:8000/predict"

# --- UI Layout ---

# Header Section
st.title("🗺️ LandCoverNet")
st.markdown("An AI-powered tool to classify land cover from satellite imagery. Simply upload an image patch from the EuroSAT dataset, and the underlying ResNet model will predict its category.")

st.markdown("---") # A nice separator

# File Uploader
uploaded_file = st.file_uploader(
    "Upload your image patch here",
    type=["jpg", "png", "tif", "tiff"]
)

# A dictionary to map class names to descriptive emojis for a friendly touch
CLASS_EMOJIS = {
    "AnnualCrop": "🌾", "Forest": "🌲", "HerbaceousVegetation": "🌿",
    "Highway": "🛣️", "Industrial": "🏭", "Pasture": "🐄",
    "PermanentCrop": "🍇", "Residential": "🏘️", "River": "💧", "SeaLake": "🌊"
}

if uploaded_file is not None:
    # Use columns for a clean side-by-side layout
    col1, col2 = st.columns([1, 1]) # Equal width columns

    with col1:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        # While waiting for the API, show a spinner
        with st.spinner('Analyzing image...'):
            files = {"file": uploaded_file.getvalue()}
            
            try:
                # Send request to the backend
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    class_name = result["predicted_class"]
                    confidence = result["confidence"]
                    
                    # Display the results in a clean format
                    st.subheader("Analysis Result")
                    
                    emoji = CLASS_EMOJIS.get(class_name, "❓")
                    st.markdown(f"### {emoji} {class_name}")
                    
                    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
                    
                    # A subtle success message
                    st.success("Classification successful!")
                    
                else:
                    st.error(f"Error from server: {response.status_code}")
                    st.write(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the API server. Please ensure the backend is running.")

# A simple footer
st.markdown("---")
st.markdown("Powered by PyTorch, FastAPI, and Streamlit.")