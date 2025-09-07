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

# ---------- CONFIG ----------
IMG_SIZE = 48
MODEL_DIR = "models"
DEFAULT_CLASS_NAMES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
# ----------------------------

def ensure_models_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

def detect_column(df, candidates):
    """Return first matching column name from candidates present in df, else None.
       Matching is case-insensitive as well.
    """
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def list_image_files_in_dir(images_dir):
    """Return list of image file paths under images_dir (non-recursive)."""
    if not os.path.isdir(images_dir):
        return []
    files = []
    for fname in os.listdir(images_dir):
        path = os.path.join(images_dir, fname)
        if os.path.isfile(path) and fname.lower().endswith(IMAGE_EXTS):
            files.append(path)
    return files

def recursive_list_images(images_dir):
    """Return list of (path, relative_parent_dirname) for all images under images_dir recursively.
       parent_dirname is the immediate folder under images_dir (or '' if image at root).
    """
    results = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                rel = os.path.relpath(root, images_dir)
                parent = '' if rel == '.' else os.path.basename(root)
                results.append((os.path.join(root, f), parent))
    return results

def try_match_setid_to_file(images_dir, set_id_value):
    """Try to find a file in images_dir whose basename starts with set_id_value or equals it (w/o ext)."""
    # fast listing
    if not os.path.isdir(images_dir):
        return None
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        name_no_ext = os.path.splitext(fname)[0]
        # match exact or startswith (e.g., '123' -> '123.jpg' or '123_1.jpg')
        if str(name_no_ext) == str(set_id_value) or str(name_no_ext).startswith(str(set_id_value) + "_") or str(name_no_ext).startswith(str(set_id_value) + "-"):
            return os.path.join(images_dir, fname)
    # second pass: recursive search
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith(IMAGE_EXTS):
                continue
            name_no_ext = os.path.splitext(f)[0]
            if str(name_no_ext) == str(set_id_value) or str(name_no_ext).startswith(str(set_id_value) + "_") or str(name_no_ext).startswith(str(set_id_value) + "-"):
                return os.path.join(root, f)
    return None

def load_data(csv_path, images_dir, image_col_override=None, label_col_override=None):
    """Load images and labels with robust fallbacks:
       - explicit overrides (--image-col / --label-col)
       - detect common image column names
       - detect label column names (emotion, label, gender, age, country)
       - if no image column, try to match set_id/id to filenames in images_dir
       - if images_dir has subfolders, use subfolder names as labels
    Returns: X, y_categorical, label_encoder, class_names
    """
    df = pd.read_csv(csv_path)
    if df is None or df.shape[0] == 0:
        raise RuntimeError(f"CSV {csv_path} empty or unreadable.")

    # Try explicit overrides first
    img_col = image_col_override if image_col_override in df.columns else None
    label_col = label_col_override if label_col_override in df.columns else None

    # Candidate lists
    img_candidates = ['image', 'filename', 'file', 'img', 'path', 'image_path', 'imageFile', 'filepath']
    label_candidates = ['emotion', 'label', 'class', 'target', 'emotion_label', 'emotionName']
    fallback_label_candidates = ['gender', 'age', 'country']

    # Expand detection with common id column names
    id_candidates = ['set_id', 'id', 'image_id', 'img_id', 'filename_id']

    # detect if not overridden
    if img_col is None:
        img_col = detect_column(df, img_candidates)
    if label_col is None:
        label_col = detect_column(df, label_candidates)

    # If still no label found, try fallback label candidates
    if label_col is None:
        label_col = detect_column(df, fallback_label_candidates)

    # If image column found: use it directly
    X_list = []
    y_list = []
    missing_files = 0

    if img_col:
        print(f"Using CSV image column: '{img_col}'")
        for idx, row in df.iterrows():
            raw = str(row[img_col])
            # try absolute path
            candidates_to_try = []
            if os.path.isabs(raw):
                candidates_to_try.append(raw)
            # try as path relative to images_dir
            candidates_to_try.append(os.path.join(images_dir, raw))
            # sometimes CSV has path-like 'subdir/file.jpg'
            candidates_to_try.append(os.path.normpath(raw))
            found = False
            for p in candidates_to_try:
                if os.path.exists(p):
                    found = True
                    img_path = p
                    break
            if not found:
                # try with common extensions appended (if raw looks like id)
                for ext in IMAGE_EXTS:
                    p = os.path.join(images_dir, raw + ext)
                    if os.path.exists(p):
                        found = True
                        img_path = p
                        break
            if not found:
                missing_files += 1
                print(f"Warning: couldn't find image for row {idx} (tried {candidates_to_try}). Skipping.")
                continue
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img, dtype=np.float32) / 255.0
                X_list.append(arr)
                # label
                if label_col and pd.notna(row[label_col]):
                    y_list.append(row[label_col])
                else:
                    y_list.append(None)
            except Exception as e:
                missing_files += 1
                print(f"Warning: error loading {img_path} (row {idx}): {e}. Skipping.")
                continue

    else:
        # No explicit image column in CSV -> attempt other fallbacks
        print("No image column found in CSV. Trying fallbacks (set_id matching and images_dir subfolders).")
        # If images_dir has subfolders, prefer that method (subfolder name -> label)
        if os.path.isdir(images_dir):
            subdirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        else:
            subdirs = []
        if subdirs:
            print(f"Detected subdirectories in images_dir; will load images and use folder names as labels. Folders: {subdirs[:10]}")
            for root, dirs, files in os.walk(images_dir):
                for f in files:
                    if not f.lower().endswith(IMAGE_EXTS):
                        continue
                    full = os.path.join(root, f)
                    parent = os.path.relpath(root, images_dir)
                    if parent == '.':
                        parent = os.path.basename(root)
                    label = '' if parent == '.' else os.path.basename(parent)
                    try:
                        img = Image.open(full).convert('L')
                        img = img.resize((IMG_SIZE, IMG_SIZE))
                        arr = np.array(img, dtype=np.float32) / 255.0
                        X_list.append(arr)
                        y_list.append(label)
                    except Exception as e:
                        missing_files += 1
                        print(f"Warning: failed to load image {full}: {e}.")
            if len(X_list) == 0:
                raise RuntimeError(f"No images found under {images_dir}.")
        else:
            # No subfolders; try matching set_id/id column in CSV to filenames in images_dir
            id_col = detect_column(df, id_candidates)
            if id_col:
                print(f"Trying to match id column '{id_col}' to image filenames in {images_dir}.")
                for idx, row in df.iterrows():
                    sid = row[id_col]
                    if pd.isna(sid):
                        missing_files += 1
                        continue
                    match = try_match_setid_to_file(images_dir, sid)
                    if not match:
                        missing_files += 1
                        print(f"Warning: no matching file for set id '{sid}' (row {idx}).")
                        continue
                    try:
                        img = Image.open(match).convert('L')
                        img = img.resize((IMG_SIZE, IMG_SIZE))
                        arr = np.array(img, dtype=np.float32) / 255.0
                        X_list.append(arr)
                        # choose label from label_col if exists, else try fallback label candidates in CSV
                        if label_col and label_col in df.columns and pd.notna(row[label_col]):
                            y_list.append(row[label_col])
                        else:
                            # try other sensible columns as labels
                            other_label = detect_column(df, fallback_label_candidates)
                            if other_label and pd.notna(row[other_label]):
                                y_list.append(row[other_label])
                            else:
                                # store None for now
                                y_list.append(None)
                    except Exception as e:
                        missing_files += 1
                        print(f"Warning: loading matched file {match} failed: {e}")
                        continue
                if len(X_list) == 0:
                    raise RuntimeError("Tried set_id matching but found no images. Check images_dir and CSV set_id values.")
            else:
                raise KeyError(
                    "Could not auto-detect image infos.\n"
                    f"CSV columns: {list(df.columns)}\n"
                    f"Images dir: {images_dir}\n"
                    "Fallbacks tried: subfolder labels, set_id matching.\n"
                    "Solution: either provide images arranged in subfolders (labelled), or add an image filename column to the CSV,\n"
                    "or pass --image-col and --label-col to the script to explicitly tell which columns to use."
                )

    # Now X_list and y_list are assembled (y_list may contain None)
    if len(X_list) == 0:
        raise RuntimeError("No images loaded after all fallbacks. Check CSV and images directory.")

    # If label_col supplied and some labels are None, try to fill using fallback columns
    # If many labels are None but folder-based labels previously produced text labels, that's fine.
    labels_filled = 0
    for i, lab in enumerate(y_list):
        if lab is None:
            labels_filled += 1
    if labels_filled > 0:
        print(f"Note: {labels_filled} entries had missing labels. If you intended to supply labels in CSV, pass --label-col.")

    X = np.array(X_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Prepare labels for encoding: If some labels are None, fill with 'unknown'
    y_safe = ['unknown' if (lab is None or (isinstance(lab, float) and np.isnan(lab))) else str(lab).strip() for lab in y_list]

    # If labels are numeric and map to DEFAULT_CLASS_NAMES range, map them
    y_series = pd.Series(y_safe)
    if y_series.dropna().shape[0] > 0 and pd.api.types.is_numeric_dtype(y_series.dropna().apply(pd.to_numeric, errors='coerce')):
        # convert numeric strings to ints where possible
        try:
            y_int = y_series.dropna().astype(float).astype(int)
            minv, maxv = int(y_int.min()), int(y_int.max())
            if minv >= 0 and maxv < len(DEFAULT_CLASS_NAMES):
                # map
                y_safe = [DEFAULT_CLASS_NAMES[int(x)] if (str(x).replace('.0','').isdigit() and int(float(x)) >= 0 and int(float(x)) < len(DEFAULT_CLASS_NAMES)) else str(x) for x in y_safe]
        except Exception:
            pass

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_safe)
    class_names = list(le.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=len(class_names))

    print(f"Loaded {len(X)} images. Missing/skipped files: {missing_files}. Classes found: {class_names}")
    return X, y_categorical, le, class_names

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=8):
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def real_time_prediction(model, class_names, camera_index=0):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
                except Exception:
                    continue
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=(0, -1))
                prediction = model.predict(roi_gray, verbose=0)
                emotion_idx = np.argmax(prediction)
                emotion = class_names[emotion_idx]
                confidence = np.max(prediction)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            cv2.imshow("Real-time Emotion Detection (press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stopping real-time prediction...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def safe_stratify_arg(y):
    """Return an array suitable for stratify or None if stratify not possible."""
    try:
        vals = np.argmax(y, axis=1) if y.ndim==2 else np.array(y)
        unique, counts = np.unique(vals, return_counts=True)
        if len(unique) < 2:
            return None
        # stratify only if every class has at least 2 examples for a train/test split
        if np.min(counts) < 2:
            return None
        return vals
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition using CNN (robust CSV handling)")
    parser.add_argument("--csv", type=str, default="emotions.csv", help="Path to emotions CSV file")
    parser.add_argument("--images", type=str, default="images", help="Path to images directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for real-time prediction")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Run real-time prediction")
    parser.add_argument("--model_path", type=str, default=os.path.join(MODEL_DIR, "emotion_model.h5"), help="Path to save/load model")
    parser.add_argument("--image-col", type=str, default=None, help="CSV column name that contains image filenames/paths (override autodetect)")
    parser.add_argument("--label-col", type=str, default=None, help="CSV column name that contains labels/emotions (override autodetect)")
    args = parser.parse_args()

    # Load and prepare data
    X, y, le, class_names = load_data(args.csv, args.images, image_col_override=args.image_col, label_col_override=args.label_col)

    # Train/test/val split (stratify when safe)
    stratify_vals = safe_stratify_arg(y)
    if stratify_vals is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_vals)
        stratify_vals2 = safe_stratify_arg(y_train)
        if stratify_vals2 is not None:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train,axis=1))
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Build or load model
    model_path = args.model_path
    if args.train or not os.path.exists(model_path):
        print("Training new model...")
        ensure_models_dir()
        num_classes = len(class_names)
        model = build_model(num_classes=num_classes)
        history = train_model(model, X_train, y_train, X_val, y_val, args.epochs, args.batch_size)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        evaluate_model(model, X_test, y_test, class_names)
    else:
        print("Loading existing model...")
        model = load_model(model_path)

    if args.predict:
        print("Starting real-time prediction. Press 'q' to quit.")
        real_time_prediction(model, class_names, args.camera)

if __name__ == "__main__":
    main()
