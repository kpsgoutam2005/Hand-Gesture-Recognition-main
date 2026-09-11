import os

class Config:
    # Directory configuration
    BASE_DIR = "/Users/harshithkethireddy/Documents/intelligent_data"
    TRAIN_DIR = os.path.join(BASE_DIR, "train/train")
    TEST_DIR = os.path.join(BASE_DIR, "test/test")
    
    # Image parameters
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 20
    GRAYSCALE = True
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    VALIDATION_RATIO = 0.2
    RANDOM_SEED = 42
    
    # Augmentation parameters
    ROTATION_RANGE = 15  # Degrees
    TRANSLATE_RANGE = 0.1  # Fraction of total width/height
    ZOOM_RANGE = 0.1  # Fraction
