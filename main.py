import io
import json
import torch
import numpy as np
import rasterio
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import models, transforms

# --- 1. CONFIGURATION AND MODEL LOADING (Runs only once on startup) ---

print("Starting server: Loading model and configuration...")

# This list must be in the correct order (0, 1, 2, ...) to match the model's output.
# The best way is to load it from the label_map.json created during training.
try:
    with open('label_map.json', 'r') as f:
        class_map = json.load(f)
        # Sort by value (0, 1, 2...) to ensure the order is correct
        CLASS_NAMES = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]
    print(f"Loaded class names: {CLASS_NAMES}")
except FileNotFoundError:
    # If the file is not found, use a hardcoded list as a fallback.
    # MAKE SURE THIS ORDER IS CORRECT!
    print("Warning: label_map.json not found. Using hardcoded class names.")
    CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
                   'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']


# Load the model architecture (ResNet18). We use weights=None because we are loading our own.
model = models.resnet18(weights=None)

# Get the number of input features for the classifier's last layer
num_ftrs = model.fc.in_features

# Replace the final layer with a new one that matches our number of classes (10)
model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))

# Load the trained weights from your .pth file.
# map_location=torch.device('cpu') ensures the model loads on a CPU-only machine.
MODEL_PATH = 'landcovernet_resnet18_best.pth'
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'. Please place it in the same directory.")
    exit() # Exit the script if the model isn't found

# Set the model to evaluation mode. This is a CRUCIAL step that disables layers like Dropout.
model.eval()

print("Model loaded successfully!")

# --- 2. DEFINE IMAGE TRANSFORMATIONS ---
# These must be the EXACT SAME as the validation transformations in your training script.
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 3. INITIALIZE FASTAPI APP ---
app = FastAPI(
    title="LandCoverNet API",
    description="An API for classifying EuroSAT land cover image patches using a ResNet18 model.",
    version="1.0.0"
)


# --- 4. DEFINE API ENDPOINTS ---

@app.get("/")
def read_root():
    """A welcome message for the API root."""
    return {"message": "Welcome to LandCoverNet. Navigate to /docs to see the API documentation."}


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="An image file (GeoTIFF, JPG, PNG) to classify.")):
    """
    Accepts an image file, preprocesses it, and returns the predicted land cover class and confidence score.
    """
    # Read the uploaded file into memory
    contents = await file.read()
    
    # Use a try-except block to handle different image types gracefully
    try:
        # First, try to open it with PIL. This works for JPG, PNG, etc.
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        # If PIL fails, it might be a GeoTIFF. Try opening with rasterio.
        try:
            with io.BytesIO(contents) as memfile:
                with rasterio.open(memfile) as dataset:
                    # Rasterio reads as (channels, height, width)
                    image_data = dataset.read()
            
            # Convert to (height, width, channels) for PIL
            image_data = np.transpose(image_data, (1, 2, 0))
            image = Image.fromarray(image_data).convert("RGB")
        except Exception as e:
            # If both fail, raise an HTTP error
            raise HTTPException(status_code=400, detail=f"Could not read the image file. Error: {e}")

    # Apply the same transformations as the validation set
    input_tensor = data_transforms(image)
    
    # The model expects a batch of images. Add a batch dimension (B, C, H, W).
    input_batch = input_tensor.unsqueeze(0)

    # Make a prediction. torch.no_grad() is a crucial optimization for inference.
    with torch.no_grad():
        output = model(input_batch)
        # Apply Softmax to convert the model's raw scores (logits) into probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get the top prediction's probability and class index
    confidence, cat_id = torch.topk(probabilities, 1)
    
    predicted_class = CLASS_NAMES[cat_id.item()]
    confidence_score = confidence.item()

    # Return the result as a JSON response
    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence_score, 4)
    }