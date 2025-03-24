import gdown
import os
import joblib

# Google Drive File ID
file_id = "1KmJQn1GT7F5f8n1v67kCkn8MvHwnUzz2"
file_url = f"https://drive.google.com/uc?id={file_id}"

# Model file path
model_path = "best_size_model.pkl"

# Download model only if it doesn't exist
if not os.path.exists(model_path):
    try:
        print("📥 Downloading model from Google Drive...")
        gdown.download(file_url, model_path, quiet=False)
    except Exception as e:
        print(f"❌ Error downloading model: {e}")

# Load the model
def load_model():
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()
print("✅ Model loaded successfully!")
