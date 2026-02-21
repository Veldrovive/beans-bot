"""
Trains and tests a model for every dataset in the datasets directory.
Folder structure:
    datasets/
        <dataset_name>/
            train/
                <label>/
                    <image_name>.<extension>
            test/
                <label>/
                    <image_name>.<extension>

Only retrains models if the file contents of the dataset have changed.
Saves the file set after training.
Caches the embeddings as a map from pixel hash to embedding and stores it in a single cache file.
This cache file can be used across training sessions since the model never changes.

File lists are stored as: <dataset_name>_train_file_list.txt and <dataset_name>_test_file_list.txt
"""

from dino_classifier import DinoFeatureExtractor, Classifier
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()
import pickle
import tqdm
from sklearn.metrics import accuracy_score, log_loss
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATASETS_DIR = Path(__file__).parent / "datasets"
MODELS_DIR = Path(__file__).parent / "models"
CACHE_DIR = Path(__file__).parent / "cache"
EMBEDDING_CACHE_FILE = CACHE_DIR / "embeddings.pkl"

def get_file_hash(file_path):
    # Choose your hash algorithm (md5, sha1, sha256)
    hasher = hashlib.md5()
    
    # Open file in binary mode
    with open(file_path, 'rb') as f:
        # Read the file in chunks (e.g., 8192 bytes = 8KB)
        while chunk := f.read(8192):
            hasher.update(chunk)
            
    return hasher.hexdigest()

def _extract_embedding(img_path: Path, feature_extractor: DinoFeatureExtractor, cache: dict[str, np.ndarray]):
    img_hash = get_file_hash(img_path)
    
    if img_hash in cache:
        return cache[img_hash]

    img = Image.open(img_path).convert("RGB")
    
    embedding = feature_extractor.extract_features(img)
    cache[img_hash] = embedding
    return embedding

def main():
    # Load or create the embedding cache
    if EMBEDDING_CACHE_FILE.exists():
        print(f"Loading embedding cache from {EMBEDDING_CACHE_FILE}...")
        with open(EMBEDDING_CACHE_FILE, "rb") as f:
            embedding_cache = pickle.load(f)
    else:
        print(f"Embedding cache not found at {EMBEDDING_CACHE_FILE}. Creating new cache.")
        embedding_cache = {}

    # Load the feature extractor
    print("Initializing DINOv3 feature extractor...")
    feature_extractor = DinoFeatureExtractor()

    dataset_paths = [p for p in DATASETS_DIR.iterdir() if p.is_dir()]
    print(f"Found {len(dataset_paths)} datasets:")
    for dataset_path in dataset_paths:
        print(f"  {dataset_path.name}")

    for dataset_path in dataset_paths:
        # TRAIN
        dataset_name = dataset_path.name
        train_path = dataset_path / "train"
        test_path = dataset_path / "test"

        labels = [p.name for p in train_path.iterdir() if p.is_dir()]
        label_to_index = {label: i for i, label in enumerate(labels)}
        print(f"\n--- Training {dataset_name} model ---")
        print(f"Found {len(labels)} labels:")
        for label in labels:
            print(f"  {label}")
        
        X_train, y_train = [], []
        last_embedding_cache_size = len(embedding_cache)
        for label in labels:
            print(f"\Extracting embeddings for {dataset_name}/{label}...")
            label_image_path = train_path / label
            image_paths = list(label_image_path.iterdir())
            for image_path in tqdm.tqdm(image_paths):
                if image_path.is_file() and image_path.stem[0] not in ["_", "."]:
                    embedding = _extract_embedding(image_path, feature_extractor, embedding_cache)
                    X_train.append(embedding.squeeze())
                    label_index = label_to_index[label]
                    y_train.append(label_index)

                    if len(embedding_cache) > last_embedding_cache_size + 4:
                        last_embedding_cache_size = len(embedding_cache)
                        with open(EMBEDDING_CACHE_FILE, "wb") as f:
                            pickle.dump(embedding_cache, f)
        if len(embedding_cache) > last_embedding_cache_size:
            last_embedding_cache_size = len(embedding_cache)
            with open(EMBEDDING_CACHE_FILE, "wb") as f:
                pickle.dump(embedding_cache, f)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print(f"Loaded {len(X_train)} embeddings for {dataset_name}.")

        classifier = Classifier()
        classifier.train(X_train, y_train, label_to_index)

        print(f"Saving {dataset_name} model...")
        classifier.save(MODELS_DIR / dataset_name)
        print(f"{dataset_name} training complete!")

        # TEST
        # Here we create a confusion matrix and annotate the test images with the model's predictions
        print(f"\n--- Testing {dataset_name} model ---")
        test_output_dir = dataset_path / "test_output"
        test_output_dir.mkdir(exist_ok=True)
        annotated_imgs_dir = test_output_dir / "annotated_imgs"
        annotated_imgs_dir.mkdir(exist_ok=True)
        
        X_test, y_test = [], []
        for label in labels:
            print(f"\Extracting test embeddings for {dataset_name}/{label}...")
            annotated_imgs_label_dir = annotated_imgs_dir / label
            annotated_imgs_label_dir.mkdir(exist_ok=True)
            label_image_path = test_path / label
            image_paths = list(label_image_path.iterdir())
            for image_path in tqdm.tqdm(image_paths):
                if image_path.is_file():
                    embedding = _extract_embedding(image_path, feature_extractor, embedding_cache)
                    embedding = embedding.squeeze()
                    X_test.append(embedding)
                    label_index = label_to_index[label]
                    y_test.append(label_index)

                    # Annotate image
                    image_prob = classifier.predict_proba([embedding])[0]
                    _annotate_image(image_path, image_prob, labels, annotated_imgs_label_dir)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        print(f"Loaded {len(X_test)} test embeddings for {dataset_name}.")

        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)
        
        print(f"Accuracy: {acc}")
        print(f"Log loss: {loss}")

        cm = confusion_matrix(y_test, y_pred, labels=range(len(labels)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax)
        plt.title(f"Test Set Confusion Matrix for {dataset_name}")
        plt.savefig(test_output_dir / "confusion_matrix.png")
        plt.close()
        

def _annotate_image(image_path: Path, image_prob: np.ndarray, labels: list[str], output_dir: Path):
    img = Image.open(image_path).convert("RGB")
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Resize to consistent width (HD)
    target_width = 1920
    h, w = cv_img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    cv_img = cv2.resize(cv_img, (target_width, new_h))

    pred_idx = np.argmax(image_prob)
    pred_label = labels[pred_idx]
    pred_confidence = image_prob[pred_idx]
    
    text_lines = [f"Prediction: {pred_label} ({pred_confidence:.2%})"]

    sorted_preds = np.argsort(image_prob)[::-1]
    for i in sorted_preds:
        label = labels[i]
        prob = image_prob[i]
        text_lines.append(f"{label}: {prob:.2%}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (0, 255, 0) # Green
    bg_color = (0, 0, 0)
    
    y0, dy = 50, 50
    for i, line in enumerate(text_lines):
        y = y0 + i * dy
        
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        
        cv2.rectangle(cv_img, (5, y - h - 10), (5 + w + 20, y + 10), bg_color, -1)
        
        cv2.putText(cv_img, line, (15, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    output_path = output_dir / f"{image_path.stem}_annotated.png"
    cv2.imwrite(str(output_path), cv_img)
    
        
        
if __name__ == "__main__":
    main()