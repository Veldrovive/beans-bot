import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
from dino_classifier import DinoFeatureExtractor, Classifier, load_or_process_data, get_data_matrices, PROJECT_ROOT

# Paths
TEST_DATA_PATH = PROJECT_ROOT / "test_test_cat_data"
PREDICT_DATA_PATH = PROJECT_ROOT / "test_data"

def evaluate_model(classifier, test_embeddings, label_to_index):
    X_test, y_test, _ = get_data_matrices(test_embeddings, label_to_index)
    
    if len(X_test) == 0:
        print("No test data found matching training labels.")
        return

    print(f"Evaluating on {len(X_test)} test samples.")
    
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob, labels=list(range(len(label_to_index))))
    
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Test Log Loss: {loss:.3f}")

    # Confusion Matrix
    index_to_label = {v: k for k, v in label_to_index.items()}
    all_indices = sorted(label_to_index.values())
    display_labels = [index_to_label[i] for i in all_indices]
    
    cm = confusion_matrix(y_test, y_pred, labels=all_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax)
    plt.title("Test Set Confusion Matrix")
    save_path = PROJECT_ROOT / 'confusion_matrix_test.png'
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def annotate_images(classifier, feature_extractor):
    if not PREDICT_DATA_PATH.exists():
        print(f"Predict data path not found: {PREDICT_DATA_PATH}")
        return

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    
    for image_path in PREDICT_DATA_PATH.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue
            
        if "_annotated" in image_path.name:
            continue

        print(f"Processing {image_path.name}...")
        
        # Load and preprocess image
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {image_path}: {e}")
            continue

        # Extract features
        embeddings = feature_extractor.extract_features(pil_image)
        
        # Predict
        probs = classifier.predict_proba(embeddings)[0]
        pred_idx = np.argmax(probs)
        pred_label = classifier.index_to_label[pred_idx]
        confidence = probs[pred_idx]

        print(f"Prediction: {pred_label} ({confidence:.2%})")

        # Annotate image using OpenCV
        # Convert PIL to OpenCV format (RGB -> BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Prepare text
        text_lines = [f"Pred: {pred_label} ({confidence:.1%})"]
        
        # Add top 3 probabilities if available
        top3_indices = np.argsort(probs)[::-1][:3]
        for i in top3_indices:
            label = classifier.index_to_label[i]
            prob = probs[i]
            text_lines.append(f"{label}: {prob:.1%}")

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0) # Green
        bg_color = (0, 0, 0)
        
        y0, dy = 30, 30
        for i, line in enumerate(text_lines):
            y = y0 + i * dy
            
            # Get text size
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(cv_image, (5, y - h - 5), (5 + w + 10, y + 5), bg_color, -1)
            
            # Draw text
            cv2.putText(cv_image, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Save annotated image
        output_path = PREDICT_DATA_PATH / f"{image_path.stem}_annotated{image_path.suffix}"
        cv2.imwrite(str(output_path), cv_image)
        print(f"Saved annotated image to {output_path}")

def main():
    print("Initializing DINOv3 feature extractor...")
    feature_extractor = DinoFeatureExtractor()
    
    print("Loading classifier...")
    classifier = Classifier()
    classifier.load()
    
    print("Loading/Processing test data...")
    test_embeddings = load_or_process_data(TEST_DATA_PATH, "cached_embeddings_test.pkl", feature_extractor)
    
    print("Evaluating model...")
    evaluate_model(classifier, test_embeddings, classifier.label_to_index)
    
    print("Annotating images...")
    annotate_images(classifier, feature_extractor)
    
    print("Testing complete!")

if __name__ == "__main__":
    main()
