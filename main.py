import gdown
import os
import joblib

# Google Drive file link (get your own link)
file_url = "https://drive.google.com/uc?id=YOUR_FILE_ID"

# Download model if not present
model_path = "best_size_model.pkl"
if not os.path.exists(model_path):
    gdown.download(file_url, model_path, quiet=False)

# Load the model
def load_model():
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()
print("✅ Model loaded successfully!")
