from dino_classifier import DinoFeatureExtractor, Classifier, load_or_process_data, get_data_matrices, PROJECT_ROOT

# Paths
TRAIN_DATA_PATH = PROJECT_ROOT / "test_cat_data"

def main():
    print("Initializing DINOv3 feature extractor...")
    feature_extractor = DinoFeatureExtractor()
    
    print("Loading/Processing training data...")
    train_embeddings = load_or_process_data(TRAIN_DATA_PATH, "cached_embeddings_train.pkl", feature_extractor)
    
    print("Preparing data for training...")
    X_train, y_train, label_to_index = get_data_matrices(train_embeddings)
    
    print("Training classifier...")
    classifier = Classifier()
    classifier.train(X_train, y_train, label_to_index)
    
    print("Saving model...")
    classifier.save()
    print("Training complete!")

if __name__ == "__main__":
    main()
