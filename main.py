import os
import gdown

# Google Drive direct download link
file_url = "https://drive.google.com/uc?id=1KmJQn1GT7F5f8n1v67kCkn8MvHwnUzz2&export=download"
model_path = "size_recommender.pkl"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    gdown.download(file_url, model_path, quiet=False)

# Load model (example)
import pickle
def load_model():
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()
print("✅ Model loaded successfully!")
