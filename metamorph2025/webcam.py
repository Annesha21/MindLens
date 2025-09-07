import os
import csv
from collections import Counter, deque
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------
# USER CONFIG — edit these
# ---------------------------
MODEL_PATH = "emotion_model.h5"     # path to your saved model
VIDEO_PATH = "D:/metamorph2025/demo .mp4"            # path to the input video
OUTPUT_PATH = "annotated_output.mp4"  # set to None to disable saving annotated video
CSV_OUTPUT = None                   # set to None to auto-generate <video_basename>_emotion_summary.csv
DISPLAY_FRAMES = False              # set True to show processing window (press 'q' to quit)
SKIP_FRAMES = 0                     # process every (SKIP_FRAMES+1)th frame (0 = process every frame)
SMOOTHING_WINDOW = 5                # majority smoothing window
MIN_FACE_SIZE = (30, 30)            # min face size for Haar detector
FACE_CASCADE_PATH = None            # None -> use OpenCV default
# ---------------------------

# default labels (update if your model uses a different ordering)
DEFAULT_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# ---------------------------
# Helper functions
# ---------------------------

def get_model_io_info(model):
    """
    Return (target_size (w,h), channels, depth, is_5d) inferred from model.input_shape.

    Supports:
      - 4D input: (batch, h, w, c) -> is_5d=False, depth=None
      - 5D input: (batch, d, h, w, c) -> is_5d=True, depth=d (may be None -> variable depth)
    Falls back to (48,48) and channels=1 if dims are None or can't be inferred.
    """
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    shape = tuple(input_shape)

    if len(shape) == 4:
        _, h, w, c = shape
        depth = None
        is_5d = False
    elif len(shape) == 5:
        _, d, h, w, c = shape
        depth = d  # may be None -> variable depth
        is_5d = True
    else:
        raise ValueError(f"Unsupported model input shape (expected 4 or 5 dims): {shape}")

    # fallback defaults
    if h is None or w is None:
        h, w = 48, 48
    if c is None:
        c = 1

    return (int(w), int(h)), int(c), depth, is_5d


def preprocess_face(face_img, target_size, channels, depth, is_5d):
    """
    Resize a face image to the model's spatial size and produce an input
    tensor matching either (1, h, w, c) or (1, depth, h, w, c).

    - face_img: BGR image from cv2 (H, W, 3)
    - target_size: (w, h)
    - channels: required #channels for the model input
    - depth: required depth (may be None for variable -> we'll use 1)
    - is_5d: whether final tensor should be 5D ((1, depth, h, w, c))
    """
    target_w, target_h = target_size
    # cv2.resize expects (width, height)
    face = cv2.resize(face_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Ensure channel axis exists
    if face.ndim == 2:
        face = np.expand_dims(face, axis=-1)  # (h, w, 1)

    h, w, c_img = face.shape

    # Convert/align channels
    if channels == 1 and c_img == 3:
        # convert BGR->GRAY
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.expand_dims(gray, axis=-1)
        c_img = 1

    if c_img == 1 and channels == 3:
        # replicate gray->3 channels
        face = np.repeat(face, 3, axis=2)
        c_img = 3

    if c_img != channels:
        if c_img < channels:
            # tile channels to reach required channel count
            repeats = channels // c_img
            rem = channels - repeats * c_img
            tiled = np.tile(face, (1, 1, repeats))
            if rem > 0:
                tiled = np.concatenate([tiled, face[:, :, :rem]], axis=2)
            face = tiled
        else:
            # trim extra channels
            face = face[:, :, :channels]

    face = face.astype("float32") / 255.0  # normalize

    # decide depth to use (if model depth is variable/None, use 1)
    depth_to_use = 1
    if is_5d and depth is not None:
        try:
            # if depth is an int-like object
            if int(depth) > 0:
                depth_to_use = int(depth)
        except Exception:
            depth_to_use = 1

    # Build final tensor
    if is_5d:
        # replicate the 2D frame across the temporal/depth axis
        frames = np.stack([face.copy() for _ in range(depth_to_use)], axis=0)  # (depth, h, w, c)
        inp = np.expand_dims(frames, axis=0)  # (1, depth, h, w, c)
    else:
        inp = np.expand_dims(face, axis=0)  # (1, h, w, c)

    return inp


# ---------------------------
# Core processing
# ---------------------------

def process_video(model_path, video_path, output_path=None,
                  csv_out=None, display=False, skip_frames=0,
                  face_cascade_path=None, labels=None, smoothing_window=5,
                  min_face_size=(30,30)):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[INFO] Loading model from: {model_path}")
    model = load_model(model_path)
    print("[INFO] Model loaded.")

    try:
        target_size, channels, depth, is_5d = get_model_io_info(model)
        depth_info = depth if depth is not None else "variable"
        print(f"[INFO] Model expects spatial size={target_size}, channels={channels}, depth={depth_info}, is_5d={is_5d}")
    except Exception as e:
        print("[WARN] Could not determine model input shape automatically:", e)
        print("[INFO] Falling back to 48x48 grayscale, depth=1.")
        target_size, channels, depth, is_5d = (48,48), 1, 1, False

    if labels is None:
        labels = DEFAULT_LABELS
    n_classes = len(labels)

    if face_cascade_path is None:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"Haar cascade not found: {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    print(f"[INFO] Using face cascade: {face_cascade_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Annotated video will be saved to: {output_path}")

    if csv_out is None:
        csv_out = os.path.splitext(os.path.basename(video_path))[0] + "_emotion_summary.csv"

    frame_results = []
    video_counts = Counter()
    smooth_queue = deque(maxlen=smoothing_window)

    frame_idx = -1
    print("[INFO] Starting frame processing...")
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        frame_idx += 1

        if skip_frames > 0 and (frame_idx % (skip_frames + 1) != 0):
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_face_size)

        if len(faces) == 0:
            frame_results.append({
                'frame': frame_idx,
                'timestamp_sec': frame_idx / fps,
                'found_face': False,
                'predicted_label': None,
                'confidence': None
            })
            cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            frame_preds = []
            for (x, y, w, h) in faces:
                pad = int(0.15 * (w+h) / 2)
                x1 = max(0, x-pad); y1 = max(0, y-pad)
                x2 = min(frame.shape[1], x + w + pad); y2 = min(frame.shape[0], y + h + pad)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                # Preprocess to match model input:
                inp = preprocess_face(face_img, target_size, channels, depth, is_5d)

                # Ensure we pass the right rank to the model:
                # - if model is 5D, inp should be (1, depth, h, w, c)
                # - if model is 4D, inp should be (1, h, w, c)
                try:
                    preds_raw = model.predict(inp, verbose=0)
                except Exception as e:
                    # Try to adapt shapes (common case: model is 5D but depth was variable and we produced 4D)
                    if is_5d and inp.ndim == 4:
                        # expand to (1,1,h,w,c)
                        inp2 = np.expand_dims(inp, axis=1)
                        # if model depth was >1 and fixed, replicate
                        if depth is not None and depth > 1:
                            inp2 = np.repeat(inp2, depth, axis=1)
                        preds_raw = model.predict(inp2, verbose=0)
                    else:
                        raise e

                preds = np.array(preds_raw).reshape(-1)
                # align preds length with labels
                if preds.shape[0] != n_classes:
                    if preds.shape[0] > n_classes:
                        preds = preds[:n_classes]
                    else:
                        preds = np.concatenate([preds, np.zeros(n_classes - preds.shape[0])])
                top_idx = int(np.argmax(preds))
                label = labels[top_idx]
                conf = float(preds[top_idx])
                frame_preds.append({'bbox': (x1,y1,x2,y2), 'label': label, 'conf': conf})

            if len(frame_preds) == 0:
                frame_results.append({
                    'frame': frame_idx,
                    'timestamp_sec': frame_idx / fps,
                    'found_face': False,
                    'predicted_label': None,
                    'confidence': None
                })
            else:
                best = max(frame_preds, key=lambda p: p['conf'])
                predicted_label = best['label']
                confidence = best['conf']
                smooth_queue.append(predicted_label)
                majority = Counter(smooth_queue).most_common(1)[0][0] if len(smooth_queue) > 0 else predicted_label

                frame_results.append({
                    'frame': frame_idx,
                    'timestamp_sec': frame_idx / fps,
                    'found_face': True,
                    'predicted_label': predicted_label,
                    'confidence': confidence
                })
                video_counts[predicted_label] += 1

                # Overlay each face
                for p in frame_preds:
                    x1,y1,x2,y2 = p['bbox']
                    lab = p['label']; conf = p['conf']
                    if lab.lower() == 'happy':
                        color = (36,255,12)
                    elif lab.lower() in ('angry','sad','disgust','fear'):
                        color = (0,0,255)
                    else:
                        color = (255,255,255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    text = f"{lab} ({conf*100:.0f}%)"
                    y_text = y1 - 10 if y1 - 10 > 10 else y1 + 20
                    cv2.putText(frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.putText(frame, f"Smoothed: {majority}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        if writer is not None:
            writer.write(frame)
        if display:
            cv2.imshow("Emotion Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Early exit requested.")
                break

    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    total_detected = sum(1 for r in frame_results if r['found_face'])
    print("[INFO] Processing completed.")
    print(f"[INFO] Sampled frames: {len(frame_results)}, Frames with face: {total_detected}")
    print("[INFO] Emotion counts (frame-level):")
    for lab, cnt in video_counts.items():
        print(f"  {lab}: {cnt}")
    if video_counts:
        top = video_counts.most_common(1)[0]
        print(f"[INFO] Most frequent: {top[0]} ({top[1]} frames)")

    # write CSV
    print(f"[INFO] Writing CSV summary to: {csv_out}")
    with open(csv_out, mode='w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['frame','timestamp_sec','found_face','predicted_label','confidence'])
        for r in frame_results:
            w.writerow([r['frame'], f"{r['timestamp_sec']:.3f}", r['found_face'], r['predicted_label'], r['confidence']])

    return {
        'frames_processed': len(frame_results),
        'frames_with_face': total_detected,
        'counts': dict(video_counts),
        'csv': os.path.abspath(csv_out),
        'annotated_video': os.path.abspath(output_path) if output_path else None
    }

# ---------------------------
# Run (edit variables above and then run this file)
# ---------------------------

if __name__ == "__main__":
    summary = process_video(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        csv_out=CSV_OUTPUT,
        display=DISPLAY_FRAMES,
        skip_frames=SKIP_FRAMES,
        face_cascade_path=FACE_CASCADE_PATH,
        labels=None,
        smoothing_window=SMOOTHING_WINDOW,
        min_face_size=tuple(MIN_FACE_SIZE)
    )
    print("[INFO] Summary:", summary)
