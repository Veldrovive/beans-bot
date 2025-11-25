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

    def save(self, models_dir=MODELS_DIR):
        if self.pipeline is None:
            raise ValueError("No model to save.")
        
        with open(models_dir / "pipeline.pkl", "wb") as f:
            pickle.dump(self.pipeline, f)
        
        # Save class names list for compatibility/simplicity
        sorted_labels = [k for k, v in sorted(self.label_to_index.items(), key=lambda item: item[1])]
        with open(models_dir / "index_to_label.pkl", "wb") as f:
            pickle.dump(sorted_labels, f)
        print(f"Model saved to {models_dir}")

    def load(self, models_dir=MODELS_DIR):
        pipeline_path = models_dir / "pipeline.pkl"
        labels_path = models_dir / "index_to_label.pkl"
        
        if not pipeline_path.exists() or not labels_path.exists():
            raise FileNotFoundError("Model files not found.")
            
        with open(pipeline_path, "rb") as f:
            self.pipeline = pickle.load(f)
            
        with open(labels_path, "rb") as f:
            class_names = pickle.load(f)
            self.index_to_label = {i: name for i, name in enumerate(class_names)}
            self.label_to_index = {name: i for i, name in enumerate(class_names)}
            
        print(f"Model loaded from {models_dir}")

def process_dataset(parent_path: Path, feature_extractor: DinoFeatureExtractor, batch_size: int = 16):
    """
    Process images in the given directory and return a dictionary of {label: [embeddings]}.
    """
    if not parent_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {parent_path}")

    photo_paths = []
    labeled_embeddings = {}

    # First pass: collect all image paths
    for sub_dir in parent_path.iterdir():
        if not sub_dir.is_dir():
            continue
        
        label = sub_dir.name
        if label not in labeled_embeddings:
            labeled_embeddings[label] = []
            
        for photo_path in sub_dir.iterdir():
            if photo_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                continue
            photo_paths.append((label, photo_path))

    # Second pass: process in batches
    for i in range(0, len(photo_paths), batch_size):
        print(f"Processing batch {i // batch_size + 1}/{len(photo_paths) // batch_size + 1} from {parent_path.name}")
        batch = photo_paths[i:i + batch_size]
        photos = [Image.open(photo_path).convert("RGB") for _, photo_path in batch]
        
        embeddings = feature_extractor.extract_features(photos)
        
        for j, (label, _) in enumerate(batch):
            labeled_embeddings[label].append(embeddings[j])

    return labeled_embeddings

def load_or_process_data(data_path: Path, cache_name: str, feature_extractor: DinoFeatureExtractor):
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Processing dataset at {data_path}")
        embeddings = process_dataset(data_path, feature_extractor)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

def get_data_matrices(labeled_embeddings, label_to_index=None):
    if label_to_index is None:
        unique_labels = sorted(list(labeled_embeddings.keys()))
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    X_list = []
    y_list = []
    
    for label, embeddings in labeled_embeddings.items():
        if label not in label_to_index:
            print(f"Warning: Skipping label '{label}' not in training set.")
            continue
        
        if not embeddings:
            continue
            
        X_list.append(np.array(embeddings))
        y_list.extend([label_to_index[label]] * len(embeddings))
        
    if not X_list:
        return np.array([]), np.array([]), label_to_index

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    
    return X, y, label_to_index
