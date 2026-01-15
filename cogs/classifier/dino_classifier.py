import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2
from huggingface_hub import login

# Constants
CKPT = "facebook/dinov3-vits16-pretrain-lvd1689m"
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

class DinoFeatureExtractor:
    def __init__(self, checkpoint=CKPT, device="mps", hf_token=None):
        self.device = device
        try:
            self.processor = AutoImageProcessor.from_pretrained(checkpoint)
            self.model = AutoModel.from_pretrained(checkpoint, dtype=torch.bfloat16, device_map=device)
        except OSError as e:
            # This means the model is gated. We need to log in to access it.
            print("Model is gated. Logging in...")
            if hf_token is not None:
                login(hf_token)
            else:
                raise e
            self.processor = AutoImageProcessor.from_pretrained(checkpoint)
            self.model = AutoModel.from_pretrained(checkpoint, dtype=torch.bfloat16, device_map=device)
        
        self.model.eval()

    def extract_features(self, images):
        """
        Extract features from a list of PIL Images or a single PIL Image.
        Returns a numpy array of embeddings.
        """
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            global_feat = outputs.pooler_output
            embeddings = global_feat.to(torch.float32).cpu().numpy()
        
        return embeddings

class Classifier:
    def __init__(self):
        self.pipeline = None
        self.label_to_index = {}
        self.index_to_label = {}

    def train(self, X, y, label_to_index):
        self.label_to_index = label_to_index
        self.index_to_label = {v: k for k, v in label_to_index.items()}
        
        print(f"Training on {len(X)} samples with {len(label_to_index)} classes.")
        
        clf = LogisticRegression(
            solver='lbfgs',
            class_weight='balanced',
            C=0.1,
            max_iter=1000,
            random_state=42
        )

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        self.pipeline.fit(X, y)
        return self.pipeline

    def predict(self, embeddings):
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded.")
        return self.pipeline.predict(embeddings)

    def predict_proba(self, embeddings):
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded.")
        return self.pipeline.predict_proba(embeddings)

    def save(self, base_path):
        if self.pipeline is None:
            raise ValueError("No model to save.")
        
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(str(base_path) + ".pkl", "wb") as f:
            pickle.dump(self.pipeline, f)
        
        # Save class names list for compatibility/simplicity
        sorted_labels = [k for k, v in sorted(self.label_to_index.items(), key=lambda item: item[1])]
        with open(str(base_path) + "_labels.pkl", "wb") as f:
            pickle.dump(sorted_labels, f)
        print(f"Model saved to {base_path}")

    def load(self, base_path):
        pipeline_path = Path(str(base_path) + ".pkl")
        labels_path = Path(str(base_path) + "_labels.pkl")
        
        if not pipeline_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Model files not found at {base_path}")
            
        with open(pipeline_path, "rb") as f:
            self.pipeline = pickle.load(f)
            
        with open(labels_path, "rb") as f:
            class_names = pickle.load(f)
            self.index_to_label = {i: name for i, name in enumerate(class_names)}
            self.label_to_index = {name: i for i, name in enumerate(class_names)}
