import streamlit as st
import requests
from PIL import Image
import io

# --- Page Configuration ---
# This must be the first Streamlit command in your script.
st.set_page_config(
    page_title="LandCoverNet Explorer",
    page_icon="🌍", # A nice globe emoji
    layout="centered" # Keep the layout clean and centered
)

# --- Custom CSS for the "Earthy Vibe" ---
# We'll inject some custom CSS to style our app.
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# You can create a style.css file, or for simplicity, we can embed it.
st.markdown("""
<style>
/* Main app background */
.main {
    background-color: #F5F5DC; /* A soft beige color */
}
/* Title style */
h1 {
    color: #4B5320; /* Army Green */
    font-family: 'Garamond', serif;
}
/* Subheader and text style */
.stMarkdown, .stFileUploader label {
    color: #3B444B; /* A dark charcoal color */
}
/* Button style */
.stButton>button {
    background-color: #556B2F; /* Dark Olive Green */
    color: white;
    border-radius: 8px;
    border: none;
}
.stButton>button:hover {
    background-color: #6B8E23; /* Olive Drab */
    color: white;
}
</style>
""", unsafe_allow_html=True)


# --- API Configuration ---
API_URL = "http://127.0.0.1:8000/predict"


# --- UI Elements ---

st.title("🌍 LandCoverNet Explorer")
st.write("Upload a satellite image patch and let our AI tell you what it sees! This tool uses a ResNet18 model to classify land cover from EuroSAT imagery.")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "png", "tif", "tiff"]
)

# A dictionary to map class names to descriptive emojis
CLASS_EMOJIS = {
    "AnnualCrop": "🌾", "Forest": "🌲", "HerbaceousVegetation": "🌿",
    "Highway": "🛣️", "Industrial": "🏭", "Pasture": "🐄",
    "PermanentCrop": "🍇", "Residential": "🏘️", "River": "💧", "SeaLake": "🌊"
}

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Uploaded Image", use_column_width=True)
    
    st.write("") # Add a little space
    st.write("Classifying...")

    # When the user uploads a file, send it to the FastAPI backend
    with st.spinner('AI is thinking...'):
        files = {"file": uploaded_file.getvalue()}
        
        try:
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                class_name = result["predicted_class"]
                confidence = result["confidence"]
                
                # Display the result with a nice big emoji
                emoji = CLASS_EMOJIS.get(class_name, "❓")
                st.success(f"## {emoji} Prediction: **{class_name}**")
                
                # Use a metric to display confidence nicely
                st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
                
            else:
                st.error(f"Error from server: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the API. Is the backend server (main.py) running?")