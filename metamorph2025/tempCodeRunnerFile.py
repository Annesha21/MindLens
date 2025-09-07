import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants
IMG_SIZE = 48
MODEL_DIR = "models"
CLASS_NAMES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

def ensure_models_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(csv_path, images_dir):
    """Load and preprocess data from CSV and images directory"""
    df = pd.read_csv(csv_path)
    
    X, y = [], []
    for idx, row in df.iterrows():
        img_path = os.path.join(images_dir, row['image'])
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            X.append(img_array)
            y.append(row['emotion'])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=len(CLASS_NAMES))
    
    return X, y_categorical, le

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=8):
    """Build the CNN model architecture"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    """Train the model with callbacks"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test, le):
    """Evaluate model and show metrics"""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def real_time_prediction(model, camera_index=0):
    """Real-time emotion prediction using webcam"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=(0, -1))
                
                prediction = model.predict(roi_gray, verbose=0)
                emotion_idx = np.argmax(prediction)
                emotion = CLASS_NAMES[emotion_idx]
                confidence = np.max(prediction)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.clear()
            ax.imshow(frame_rgb)
            ax.axis('off')
            ax.set_title("Real-time Emotion Detection")
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping real-time prediction...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition using CNN")
    parser.add_argument("--csv", type=str, default="emotions.csv", help="Path to emotions CSV file")
    parser.add_argument("--images", type=str, default="images", help="Path to images directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for real-time prediction")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Run real-time prediction")
    args = parser.parse_args()

    # Load and prepare data
    X, y, le = load_data(args.csv, args.images)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Build or load model
    model_path = os.path.join(MODEL_DIR, "emotion_model.h5")
    if args.train or not os.path.exists(model_path):
        print("Training new model...")
        ensure_models_dir()
        model = build_model()
        history = train_model(model, X_train, y_train, X_val, y_val, args.epochs, args.batch_size)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, le)
    else:
        print("Loading existing model...")
        model = load_model(model_path)

    # Real-time prediction
    if args.predict:
        print("Starting real-time prediction. Press 'q' to quit.")
        real_time_prediction(model, args.camera)

if __name__ == "__main__":
    main()