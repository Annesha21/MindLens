"""
main.py

Train from images/<label>/*.jpg (no pre-existing .h5 required) and run real-time emotion detection.
Behavior:
 - If a model exists it will load and run inference.
 - If no model exists it will automatically train from images/ and save a timestamped .h5 in ./models/,
   then run inference.

Usage examples:
  - Auto-train-if-needed and run: python main.py
  - Force training from images: python main.py --train --images images --epochs 15
  - Use a specific model file: python main.py --model models/my_model.h5
"""
import os
import sys
import argparse
import time
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Try to import mediapipe for face detection (no external cascade files required)
USE_MEDIAPIPE = True
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
except Exception:
    USE_MEDIAPIPE = False

# -----------------------
# Config (no hardcoded existing .h5 requirement)
# -----------------------
IMG_SIZE = 48
MODEL_DIR = "models"
DEFAULT_MODEL_NAME = "emotion_model.h5"            # convenience "latest" name (created after training)
CLASS_NAMES_FILE = os.path.join(MODEL_DIR, "class_names.txt")
CONFUSION_MATRIX_FILE = os.path.join(MODEL_DIR, "confusion_matrix.png")

# Typical FER2013 mapping (common)
FER_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# -----------------------
# CSV mapping loader
# -----------------------
def read_csv_label_map(csv_path: str) -> Dict[str, str]:
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Warning: can't read CSV {csv_path}: {e}")
        return mapping

    cols = [c.lower() for c in df.columns]
    if 'filename' in cols and ('emotion' in cols or 'label' in cols):
        filename_col = df.columns[cols.index('filename')]
        label_col = df.columns[cols.index('emotion')] if 'emotion' in cols else df.columns[cols.index('label')]
        for _, row in df.iterrows():
            fn = str(row[filename_col]).strip()
            lab = str(row[label_col]).strip()
            if fn and lab:
                mapping[fn] = lab
    else:
        # fallback: if csv has two columns assume first is filename second is label
        if len(df.columns) >= 2:
            for _, row in df.iterrows():
                fn = str(row.iloc[0]).strip()
                lab = str(row.iloc[1]).strip()
                if fn and lab:
                    mapping[fn] = lab
    return mapping

# -----------------------
# Image loader (folder structure)
# -----------------------
def load_images_from_folders(images_dir: str, csv_map: dict = None, img_size: int = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    samples = []
    for root, _, files in os.walk(images_dir):
        rel_root = os.path.relpath(root, images_dir)
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                fullpath = os.path.join(root, fname)
                label = None
                if csv_map:
                    rpath = os.path.relpath(fullpath, images_dir)
                    # try absolute, relative and basename matches
                    if fullpath in csv_map:
                        label = csv_map[fullpath]
                    elif rpath in csv_map:
                        label = csv_map[rpath]
                    elif fname in csv_map:
                        label = csv_map[fname]
                if label is None:
                    if rel_root == ".":
                        label = os.path.basename(images_dir) or "0"
                    else:
                        label = rel_root.replace(os.sep, "_")
                samples.append((fullpath, label))

    if len(samples) == 0:
        raise ValueError("No images found in images directory.")

    labels_all = [lab for (_, lab) in samples]
    le = LabelEncoder()
    le.fit(labels_all)
    class_names = list(le.classes_)
    print("Detected classes:", class_names)

    X_list, y_list = [], []
    skipped = 0
    for path, lab in samples:
        try:
            img = Image.open(path).convert("L")
            img = img.resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X_list.append(arr)
            y_list.append(lab)
        except Exception as e:
            skipped += 1
            print(f"Warning: could not load {path}: {e}")
    if skipped:
        print(f"Skipped {skipped} images that failed to load.")

    X = np.array(X_list)
    X = X.reshape((-1, img_size, img_size, 1))
    y_enc = le.transform(y_list)
    y_cat = to_categorical(y_enc, num_classes=len(class_names))
    print("Loaded dataset shapes:", X.shape, y_cat.shape)
    return X, y_cat, class_names

# -----------------------
# Model builder
# -----------------------
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=2) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------
# Confusion matrix plotting
# -----------------------
def plot_and_save_confusion_matrix(y_true_idx, y_pred_idx, display_names, out_path=CONFUSION_MATRIX_FILE, normalize=False):
    """
    Plot & save confusion matrix. y_true_idx and y_pred_idx are integer label arrays.
    display_names: list of labels to show on the axes (in order of index).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(display_names))))
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    figsize = (max(6, len(display_names) * 0.4), max(6, len(display_names) * 0.4))
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, xticks_rotation=45, include_values=True)
    plt.title("Confusion Matrix (validation)")
    plt.tight_layout()
    try:
        fig.savefig(out_path)
        print(f"Saved confusion matrix to {out_path}")
    except Exception as e:
        print(f"Warning: failed to save confusion matrix to {out_path}: {e}")
    plt.show(block=False)
    plt.pause(0.5)

# -----------------------
# Map class_names -> display (emotion) names
# -----------------------
def build_display_names(class_names: List[str]) -> List[str]:
    """
    Create friendly display names (emotion words) for each class index.
    Rules:
      - If a class_name already contains alphabetic characters (e.g., 'happy'), use it.
      - Else, if classes are numeric strings and form 0..6, map to FER_EMOTIONS.
      - Else, keep original class_name.
    """
    # if already words, return them lowercased (clean)
    alpha_names = [cn for cn in class_names if any(c.isalpha() for c in cn)]
    if len(alpha_names) == len(class_names):
        # all are "word-like" -> normalize casing
        return [cn.lower() for cn in class_names]

    # try numeric detection
    numeric_flags = []
    numeric_values = []
    for cn in class_names:
        try:
            v = int(cn)
            numeric_flags.append(True)
            numeric_values.append(v)
        except Exception:
            numeric_flags.append(False)
            numeric_values.append(None)

    # all numeric and within 0..6 -> map to FER_EMOTIONS
    if all(numeric_flags):
        max_v = max([v for v in numeric_values if v is not None])
        min_v = min([v for v in numeric_values if v is not None])
        if 0 <= min_v and max_v <= 6:
            # map numeric index to FER list (handles unordered class_names as well)
            # create mapping index->FER label by matching index positions
            # class_names may be like ['0','1','2'] but LabelEncoder ensured index order matches class_names order
            mapped = []
            for cn in class_names:
                idx = int(cn)
                label = FER_EMOTIONS[idx] if idx < len(FER_EMOTIONS) else cn
                mapped.append(label)
            return mapped
        else:
            # numeric but outside 0..6 -> try partial mapping for 0..6 and leave others numeric
            mapped = []
            for cn in class_names:
                try:
                    idx = int(cn)
                    if 0 <= idx < len(FER_EMOTIONS):
                        mapped.append(FER_EMOTIONS[idx])
                    else:
                        mapped.append(cn)
                except Exception:
                    mapped.append(cn)
            return mapped

    # mixed names: use alphabetic ones lowercased and keep numeric as-is
    out = []
    for cn in class_names:
        if any(c.isalpha() for c in cn):
            out.append(cn.lower())
        else:
            out.append(cn)
    return out

# -----------------------
# Face detection + inference (uses mediapipe if available)
# -----------------------
def detect_faces_mediapipe(rgb_frame, mp_detector) -> List[tuple]:
    h, w, _ = rgb_frame.shape
    results = mp_detector.process(rgb_frame)
    boxes = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = int(max(0, bbox.xmin) * w)
            y1 = int(max(0, bbox.ymin) * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            x2 = x1 + bw
            y2 = y1 + bh
            x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
            boxes.append((x1, y1, x2, y2))
    return boxes

def detect_and_predict(frame_bgr, model, display_names, mp_detector=None) -> Tuple[np.ndarray, str]:
    """
    Uses display_names (friendly labels) for overlay.
    """
    h, w = frame_bgr.shape[:2]
    predicted_label = "No face"
    face_crop = None

    if mp_detector is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = detect_faces_mediapipe(rgb, mp_detector)
        if len(boxes) > 0:
            boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            x1, y1, x2, y2 = boxes[0]
            pad_x = int(0.15 * (x2 - x1)); pad_y = int(0.15 * (y2 - y1))
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(w-1, x2 + pad_x); y2 = min(h-1, y2 + pad_y)
            face_crop = frame_bgr[y1:y2, x1:x2]
            cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)

    if face_crop is None:
        # fallback: whole frame -> convert to grayscale and resize
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        try:
            face_resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        except Exception:
            return frame_bgr, "ResizeError"
        face_resized = face_resized.astype('float32') / 255.0
        face_resized = np.expand_dims(face_resized, axis=(0, -1))
        preds = model.predict(face_resized, verbose=0)
        idx = int(np.argmax(preds[0])); conf = float(preds[0][idx])
        label = display_names[idx] if idx < len(display_names) else str(idx)
        predicted_label = f"{label} ({conf*100:0.1f}%)"
        cv2.putText(frame_bgr, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return frame_bgr, predicted_label

    # process detected face crop
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    try:
        face_resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    except Exception:
        return frame_bgr, "ResizeError"
    face_resized = face_resized.astype('float32') / 255.0
    face_resized = np.expand_dims(face_resized, axis=(0, -1))
    preds = model.predict(face_resized, verbose=0)
    idx = int(np.argmax(preds[0])); conf = float(preds[0][idx])
    label = display_names[idx] if idx < len(display_names) else str(idx)
    predicted_label = f"{label} ({conf*100:0.1f}%)"
    cv2.putText(frame_bgr, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return frame_bgr, predicted_label

# -----------------------
# Camera loop (matplotlib)
# -----------------------
def run_camera(model, display_names, cam_index=0):
    mp_detector = None
    if USE_MEDIAPIPE:
        mp_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        print("Using MediaPipe face detection.")
    else:
        print("MediaPipe not available — predictions will be run on the whole frame as fallback.")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try another --cam index or check permissions.")

    plt.ion()
    fig, ax = plt.subplots(figsize=(6,5))
    im = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            annotated, label = detect_and_predict(frame, model, display_names, mp_detector=mp_detector)
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            if im is None:
                im = ax.imshow(frame_rgb)
                ax.axis('off')
                ax.set_title(label)
            else:
                im.set_data(frame_rgb)
                ax.set_title(label)
            fig.canvas.draw_idle()
            plt.pause(0.001)

            if not plt.fignum_exists(fig.number):
                print("Matplotlib window closed. Exiting.")
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        plt.close(fig)
        if USE_MEDIAPIPE and mp_detector is not None:
            mp_detector.close()
        print("Camera released and windows closed.")

# -----------------------
# Save & load class names
# -----------------------
def save_class_names(class_names: List[str], filepath: str = CLASS_NAMES_FILE):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")
    print(f"Saved {len(class_names)} class names to {filepath}")

def load_class_names(filepath: str = CLASS_NAMES_FILE) -> List[str]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# -----------------------
# Utility: decide model path & train-if-needed
# -----------------------
def ensure_models_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

def choose_model_path(explicit_model: str = None) -> str:
    ensure_models_dir()
    if explicit_model:
        return explicit_model
    # prefer existing "latest" default if present
    latest = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)
    if os.path.exists(latest):
        return latest
    # else create timestamped filename for new training
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(MODEL_DIR, f"emotion_model_{ts}.h5")

# -----------------------
# Main CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Train + run real-time emotion detector (no pre-existing .h5 required).")
    parser.add_argument("--train", action="store_true", help="Force training from images folder and save model.")
    parser.add_argument("--images", type=str, default="images", help="Images folder (default 'images').")
    parser.add_argument("--csv", type=str, default="emotions.csv", help="Optional CSV with filename->label mappings.")
    parser.add_argument("--model", type=str, default=None, help="Optional model path to load/save.")
    parser.add_argument("--cam", type=int, default=0, help="Camera index.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default 50).")
    parser.add_argument("--batch", type=int, default=64, help="Training batch size (default 64).")
    parser.add_argument("--normalize-cm", action="store_true", help="Normalize confusion matrix rows.")
    args = parser.parse_args()

    csv_map = {}
    if args.csv and os.path.exists(args.csv):
        csv_map = read_csv_label_map(args.csv)
        if csv_map:
            print(f"Loaded CSV mapping for {len(csv_map)} filenames from {args.csv}")
        else:
            print(f"No usable mapping in {args.csv}; proceeding with folder labels.")

    model_path = choose_model_path(args.model)

    # If user explicitly asked to train -> train and save to chosen model_path
    if args.train:
        print("Training requested by user.")
        X, y, class_names = load_images_from_folders(args.images, csv_map, img_size=IMG_SIZE)
        num_classes = y.shape[1]
        model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=num_classes)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        print("Starting training...")
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch, verbose=1)

        # build friendly display names
        display_names = build_display_names(class_names)

        # Compute confusion matrix on validation set
        try:
            preds_val = model.predict(X_val, verbose=0)
            y_true_idx = np.argmax(y_val, axis=1)
            y_pred_idx = np.argmax(preds_val, axis=1)
            plot_and_save_confusion_matrix(y_true_idx, y_pred_idx, display_names, out_path=CONFUSION_MATRIX_FILE, normalize=args.normalize_cm)
        except Exception as e:
            print("Warning: failed to compute or plot confusion matrix:", e)

        # save both timestamped model and a 'latest' symlink-like copy
        ensure_models_dir()
        model.save(model_path)
        # also copy to models/emotion_model.h5 for convenience
        latest = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)
        try:
            import shutil
            shutil.copy(model_path, latest)
        except Exception:
            pass
        save_class_names(class_names)
        print(f"Saved model to {model_path} and latest copy {latest}")
        run_camera(model, display_names, cam_index=args.cam)
        return

    # Not forcing train: try to load existing model; if none found, attempt automatic training
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        try:
            model = load_model(model_path)
        except Exception as e:
            print("Failed to load model (will try to train):", e)
            model = None
    else:
        model = None

    if model is None:
        # No model available -> attempt to auto-train from images directory
        if os.path.exists(args.images):
            print("No pre-existing model found. Will train a model from your images folder now.")
            X, y, class_names = load_images_from_folders(args.images, csv_map, img_size=IMG_SIZE)
            num_classes = y.shape[1]
            model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=num_classes)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
            print("Starting training...")
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch, verbose=1)

            # build friendly display names
            display_names = build_display_names(class_names)

            # Compute confusion matrix on validation set
            try:
                preds_val = model.predict(X_val, verbose=0)
                y_true_idx = np.argmax(y_val, axis=1)
                y_pred_idx = np.argmax(preds_val, axis=1)
                plot_and_save_confusion_matrix(y_true_idx, y_pred_idx, display_names, out_path=CONFUSION_MATRIX_FILE, normalize=args.normalize_cm)
            except Exception as e:
                print("Warning: failed to compute or plot confusion matrix:", e)

            # save model
            ensure_models_dir()
            model.save(model_path)
            latest = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)
            try:
                import shutil
                shutil.copy(model_path, latest)
            except Exception:
                pass
            save_class_names(class_names)
            print(f"Training complete. Model saved to {model_path} and latest copy {latest}")
        else:
            print("No model present and images folder not found. Please create 'images/' with subfolders per class,")
            print("then re-run with --train or simply run this script to auto-train.")
            sys.exit(1)
    else:
        # model loaded successfully; load class names if available
        class_names = load_class_names()
        if not class_names:
            out_shape = getattr(model, "output_shape", None)
            if isinstance(out_shape, tuple):
                num_classes = out_shape[-1]
                class_names = [str(i) for i in range(num_classes)]
            else:
                class_names = ["0"]
        # build display names from class_names
        display_names = build_display_names(class_names)

    print("Starting webcam inference (press Ctrl+C or close the window to stop).")
    print("Display labels (index -> name):")
    for i, name in enumerate(display_names):
        print(f"  {i}: {name}")
    run_camera(model, display_names, cam_index=args.cam)

if __name__ == "__main__":
    main()
