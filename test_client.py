# test_client.py
import requests

# URL of your running API
API_URL = "http://127.0.0.1:8000/predict"

# Path to a sample image you want to test
IMAGE_PATH = "test_image.jpg" # Make sure you have a test image in your folder

# Open the file in binary mode
with open(IMAGE_PATH, "rb") as image_file:
    files = {"file": (IMAGE_PATH, image_file, "image/jpeg")}
    
    try:
        # Send the POST request
        response = requests.post(API_URL, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("✅ Request successful!")
            print("Response JSON:", response.json())
        else:
            print(f"❌ Error: Received status code {response.status_code}")
            print("Response text:", response.text)
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: Could not connect to the server at {API_URL}.")
        print("   Is the uvicorn server running?")