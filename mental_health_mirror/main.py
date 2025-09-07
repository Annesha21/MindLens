import threading
import queue
import time
import sys

import cv2
import numpy as np
from deepface import DeepFace

import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk

import speech_recognition as sr
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize once (not inside the loop)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Keep conversation history
chat_history_ids = None


# -----------------------------
# Settings
# -----------------------------
CAM_INDEX = 0
ANALYZE_EVERY = 10        # run DeepFace every N frames (perf)
DRAW_CONFIDENCE = False   # set True to show confidence score
FRAME_WIDTH = 800         # display width in UI (keeps aspect ratio)

# -----------------------------
# Global state (thread flags & queues)
# -----------------------------
running = False
video_thread = None
audio_thread = None

# Queues to pass info to UI thread
emotion_queue = queue.Queue(maxsize=1)     # (results_list, timestamp)
frame_queue = queue.Queue(maxsize=1)       # latest BGR frame
stt_queue = queue.Queue(maxsize=1)         # (text, sentiment_label, score, timestamp)

# -----------------------------
# Helpers: mood fusion + suggestion
# -----------------------------
def fuse_mood(face_emotions, voice_sentiment):
    """
    face_emotions: list of dominant emotions (e.g., ["happy", "neutral", "sad"])
    voice_sentiment: "POSITIVE" | "NEGATIVE" | "NEUTRAL" (we'll map from HF)
    """
    face = None
    if face_emotions:
        counts = {}
        for e in face_emotions:
            counts[e] = counts.get(e, 0) + 1
        face = max(counts, key=counts.get)

    if voice_sentiment is None:
        v = "NEUTRAL"
    else:
        v = voice_sentiment.upper()

    if face in ["happy", "neutral"] and v == "POSITIVE":
        return "calm ☀️"
    if face in ["sad", "angry", "fear", "disgust"] or v == "NEGATIVE":
        return "stressed 🌩️"
    return "uncertain 🌫️"

def suggestion_for(mood):
    if mood == "calm ☀️":
        return "Try a gratitude minute or keep your flow. Maybe soft music 🎵"
    if mood == "stressed 🌩️":
        return "Try 4-7-8 breathing for 1–2 minutes 🧘, or a 2-min body scan."
    return "Take a short break, hydrate 💧, and do 5 slow breaths."

# -----------------------------
# Video worker (DeepFace every N frames)
# -----------------------------
def video_worker():
    global running
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not open camera index", CAM_INDEX)
        running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    last_results = []

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # latest frame (non-blocking replace)
        try:
            if frame_queue.full():
                _ = frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put(frame)

        # analyze on schedule
        if frame_count % ANALYZE_EVERY == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                if not isinstance(results, list):
                    results = [results]
                last_results = results
            except Exception:
                last_results = []

            try:
                if emotion_queue.full():
                    _ = emotion_queue.get_nowait()
            except queue.Empty:
                pass
            emotion_queue.put((last_results, time.time()))

        frame_count += 1

    cap.release()


# -----------------------------
# Audio worker (loop listen → STT → sentiment)
# -----------------------------

def audio_worker():
    global running
    recognizer = sr.Recognizer()

    # instantiate sentiment pipeline once inside thread to avoid blocking main thread on startup
    try:
        sentiment = pipeline("sentiment-analysis")
    except Exception as e:
        print("⚠️ Could not load sentiment pipeline:", e)
        sentiment = None

    try:
        mic = sr.Microphone()  # default mic
    except Exception as e:
        print("❌ Microphone init failed:", e)
        return

    while running:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            text = ""
            try:
                text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                text = ""
            except sr.RequestError:
                text = ""

            label = None
            score = None
            if text and sentiment is not None:
                try:
                    res = sentiment(text)[0]
                    label = res.get("label", None)
                    score = float(res.get("score", 0.0))
                except Exception:
                    label = None
                    score = None

            # push to queue (replace old)
            try:
                if stt_queue.full():
                    _ = stt_queue.get_nowait()
            except queue.Empty:
                pass
            stt_queue.put((text, label, score, time.time()))

        except sr.WaitTimeoutError:
            continue
        except Exception:
            # small sleep to avoid busy loop on persistent errors
            time.sleep(0.3)
            continue

# -----------------------------
# UI (Tkinter) - improved & colorful
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🪞 Mental Health Mirror (Prototype)")
        self.protocol("WM_DELETE_WINDOW", self.on_quit)

        # Layout and colors
        self.bg_main = "#f4f7fb"
        self.bg_panel = "#eef2f7"
        self.btn_start_color = "#2ecc71"
        self.btn_stop_color = "#e74c3c"
        self.btn_disabled_color = "#95a5a6"
        self.configure(bg=self.bg_main)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)

        # Left: video canvas
        self.video_label = tk.Label(self, bg="black", relief="sunken", bd=2)
        self.video_label.grid(row=0, column=0, rowspan=9, sticky="nsew", padx=8, pady=8)

        # Right panel
        right_frame = tk.Frame(self, bg=self.bg_panel)
        right_frame.grid(row=0, column=1, rowspan=9, sticky="nsew", padx=8, pady=8)
        right_frame.columnconfigure(0, weight=1)

        # Buttons
        self.btn_start = tk.Button(
            right_frame, text="▶ Start", command=self.start_all,
            bg=self.btn_start_color, fg="white", activebackground="#28b765",
            font=("Segoe UI", 11, "bold"), relief="flat", padx=10, pady=6
        )
        self.btn_start.grid(row=0, column=0, sticky="ew", pady=(0,6))

        self.btn_stop = tk.Button(
            right_frame, text="⏹ Stop", command=self.stop_all,
            state=tk.DISABLED, bg=self.btn_disabled_color, fg="white",
            activebackground=self.btn_disabled_color, font=("Segoe UI", 11, "bold"),
            relief="flat", padx=10, pady=6
        )
        self.btn_stop.grid(row=1, column=0, sticky="ew", pady=6)

        self.btn_quit = tk.Button(
            right_frame, text="❌ Quit", command=self.on_quit,
            bg="#7f8c8d", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=10, pady=6
        )
        self.btn_quit.grid(row=2, column=0, sticky="ew", pady=6)

        self.btn_video_upload = tk.Button(
            right_frame, text="🎬 Analyze Video", command=self.analyze_video_file,
            bg="#3498db", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=10, pady=6
        )
        self.btn_video_upload.grid(row=9, column=0, sticky="ew", pady=6)

        self.btn_audio_upload = tk.Button(
            right_frame, text="🎤 Analyze Audio", command=self.analyze_audio_file,
            bg="#9b59b6", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=10, pady=6
        )
        self.btn_audio_upload.grid(row=10, column=0, sticky="ew", pady=6)


        # Separator (thin line)
        tk.Frame(right_frame, bg="#d0d7de", height=2).grid(row=3, column=0, sticky="ew", pady=10)

        # Live text variables and widgets
        self.emotion_var = tk.StringVar(value="Face emotions: —")
        self.transcript_var = tk.StringVar(value="Transcript: —")
        self.sentiment_var = tk.StringVar(value="Voice: —")
        self.mood_var = tk.StringVar(value="Mood: —")
        self.suggestion_var = tk.StringVar(value="Suggestion: —")

        label_style = {"anchor":"w", "justify":"left", "bg":self.bg_panel, "fg":"#2c3e50", "font":("Segoe UI", 11)}

        self.lbl_emotion = tk.Label(right_frame, textvariable=self.emotion_var, wraplength=320, **label_style)
        self.lbl_emotion.grid(row=4, column=0, sticky="ew", pady=4)

        self.lbl_transcript = tk.Label(right_frame, textvariable=self.transcript_var, wraplength=320, **label_style)
        self.lbl_transcript.grid(row=5, column=0, sticky="ew", pady=4)

        self.lbl_sentiment = tk.Label(right_frame, textvariable=self.sentiment_var, wraplength=320, **label_style)
        self.lbl_sentiment.grid(row=6, column=0, sticky="ew", pady=4)

        # mood label (store widget to change color)
        self.lbl_mood = tk.Label(right_frame, textvariable=self.mood_var,
                                 font=("Segoe UI", 13, "bold"), fg="#34495e", bg=self.bg_panel)
        self.lbl_mood.grid(row=7, column=0, sticky="ew", pady=(6,2))

        self.lbl_suggestion = tk.Label(
            right_frame, textvariable=self.suggestion_var, wraplength=320,
            anchor="w", justify="left", font=("Segoe UI", 11, "italic"),
            bg=self.bg_panel, fg="#2c3e50"
        )
        self.lbl_suggestion.grid(row=8, column=0, sticky="ew", pady=(2,8))

        # Conversation log
        self.chat_log = tk.Text(right_frame, height=12, wrap="word", font=("Segoe UI", 10))
        self.chat_log.grid(row=11, column=0, sticky="nsew", pady=6)
        self.chat_log.insert("end", "🤖 AI: Hello! How are you feeling today?\n")
        self.chat_log.config(state=tk.DISABLED)

        # Button for real-time AI talk
        self.btn_ai_talk = tk.Button(
            right_frame, text="🎙️ AI Conversation", command=self.start_ai_conversation,
            bg="#e67e22", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=10, pady=6
        )
        self.btn_ai_talk.grid(row=12, column=0, sticky="ew", pady=6)


        self.btn_analyze_chat = tk.Button(
            right_frame, text="🧠 Analyze Chat Mood", command=self.analyze_chat_text,
            bg="#16a085", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=10, pady=6
        )
        self.btn_analyze_chat.grid(row=13, column=0, sticky="ew", pady=6)


        # internal state
        self.latest_results = []
        self.latest_text = ""
        self.latest_sentiment = None
        self.latest_score = None

        # start UI updater
        self.after(30, self.ui_loop)

    # -------------------------
    # Controls
    # -------------------------
    def start_all(self):
        global running, video_thread, audio_thread
        if running:
            return
        running = True

        # button visual feedback
        self.btn_start.config(state=tk.DISABLED, bg=self.btn_disabled_color)
        self.btn_stop.config(state=tk.NORMAL, bg=self.btn_stop_color)

        # spawn workers
        video_thread = threading.Thread(target=video_worker, daemon=True)
        video_thread.start()
        audio_thread = threading.Thread(target=audio_worker, daemon=True)
        audio_thread.start()

    def stop_all(self):
        global running
        running = False
        # button visual feedback
        self.btn_start.config(state=tk.NORMAL, bg=self.btn_start_color)
        self.btn_stop.config(state=tk.DISABLED, bg=self.btn_disabled_color)

    def on_quit(self):
        self.stop_all()
        # ensure threads have a chance to exit gracefully
        time.sleep(0.25)
        self.destroy()

    # -------------------------
    # Drawing & UI loop
    # -------------------------
    def draw_emotions_on_frame(self, frame_bgr, results):
        """Draw rectangles + emotion label on BGR frame with color per emotion."""
        if not results:
            return frame_bgr

        # emotion -> BGR color
        colors = {
            "happy": (60, 180, 75),
            "sad": (66, 133, 244),
            "angry": (220, 53, 69),
            "neutral": (120, 120, 120),
            "surprise": (255, 193, 7),
            "fear": (128, 0, 128),
            "disgust": (0, 128, 128)
        }

        for res in results:
            region = res.get("region", None)
            dom = (res.get("dominant_emotion") or "Unknown").lower()
            if region:
                x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                color = colors.get(dom, (0, 255, 0))
                # OpenCV uses BGR
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
                label = dom.capitalize()
                if DRAW_CONFIDENCE:
                    try:
                        confs = res.get("emotion", {})
                        if dom in confs:
                            label = f"{label} {confs[dom]:.2f}"
                    except Exception:
                        pass
                cv2.putText(frame_bgr, label, (x, max(y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame_bgr

    def ui_loop(self):
        """Pull latest data from queues, update canvas + labels."""
        # 1) Get latest frame
        frame = None
        try:
            while True:
                frame = frame_queue.get_nowait()
        except queue.Empty:
            pass

        # 2) Get latest emotion results
        try:
            while True:
                self.latest_results, _ = emotion_queue.get_nowait()
        except queue.Empty:
            pass

        # 3) Get latest STT + sentiment
        try:
            while True:
                self.latest_text, self.latest_sentiment, self.latest_score, _ = stt_queue.get_nowait()
        except queue.Empty:
            pass

        # Update labels
        face_emotions = [res.get("dominant_emotion", "Unknown") for res in (self.latest_results or [])]
        if face_emotions:
            self.emotion_var.set(f"Face emotions: {', '.join(face_emotions)}")
        else:
            self.emotion_var.set("Face emotions: —")

        if self.latest_text:
            self.transcript_var.set(f"Transcript: {self.latest_text}")
        else:
            self.transcript_var.set("Transcript: —")

        if self.latest_sentiment:
            self.sentiment_var.set(f"Voice: {self.latest_sentiment} ({(self.latest_score or 0):.2f})")
        else:
            self.sentiment_var.set("Voice: —")

        # Fuse mood & set color
        mood = fuse_mood(face_emotions, self.latest_sentiment)
        self.mood_var.set(f"Mood: {mood}")

        mood_color_map = {
            "calm ☀️": "#27ae60",
            "stressed 🌩️": "#e74c3c",
            "uncertain 🌫️": "#f39c12"
        }
        self.lbl_mood.config(fg=mood_color_map.get(mood, "#34495e"))
        self.suggestion_var.set(f"Suggestion: {suggestion_for(mood)}")

        # Draw & show the frame
        if frame is not None:
            frame = self.draw_emotions_on_frame(frame, self.latest_results)

            # Resize for UI width
            h, w = frame.shape[:2]
            scale = FRAME_WIDTH / float(w)
            new_w = FRAME_WIDTH
            new_h = int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))

            # Convert BGR->RGB for Tkinter
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

        # schedule next update
        self.after(30, self.ui_loop)

    def analyze_video_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not filepath:
            return

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                results = DeepFace.analyze(
                    frame, actions=["emotion"], enforce_detection=False, detector_backend="opencv"
                )
                if not isinstance(results, list):
                    results = [results]
            except Exception:
                results = []

            frame = self.draw_emotions_on_frame(frame, results)

            # Resize for display
            h, w = frame.shape[:2]
            scale = FRAME_WIDTH / float(w)
            new_w, new_h = FRAME_WIDTH, int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

            self.update()
        cap.release()

    def analyze_audio_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Audio",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac")]
        )
        if not filepath:
            return

        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(filepath) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
        except Exception as e:
            messagebox.showerror("Error", f"Speech recognition failed: {e}")
            return

        try:
            sentiment = pipeline("sentiment-analysis")
            res = sentiment(text)[0]
            label, score = res.get("label"), float(res.get("score"))
        except Exception as e:
            label, score = "Unknown", 0.0

        # update UI directly
        self.transcript_var.set(f"Transcript: {text}")
        self.sentiment_var.set(f"Voice: {label} ({score:.2f})")

        mood = fuse_mood([], label)   # only voice here
        self.mood_var.set(f"Mood: {mood}")
        self.lbl_mood.config(fg={"calm ☀️":"#27ae60","stressed 🌩️":"#e74c3c","uncertain 🌫️":"#f39c12"}.get(mood,"#34495e"))
        self.suggestion_var.set(f"Suggestion: {suggestion_for(mood)}")

    def start_ai_conversation(self):
        global running
        if running:  # stop normal live mode if running
            self.stop_all()
        threading.Thread(target=self.ai_conversation_worker, daemon=True).start()

    def ai_conversation_worker(self):

        recognizer = sr.Recognizer()
        try:
            mic = sr.Microphone()
        except Exception as e:
            messagebox.showerror("Error", f"Microphone not found: {e}")
            return

        try:
            sentiment = pipeline("sentiment-analysis")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        except Exception as e:
            messagebox.showerror("Error", f"AI pipeline failed: {e}")
            return

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        self.append_chat("🤖", "Conversation started. Speak when ready!")

        chat_history_ids = None  # keeps track of dialogue context

        while True:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
                text = recognizer.recognize_google(audio)

                if not text:
                    continue

                # --- Mood analysis ---
                res = sentiment(text)[0]
                label, score = res["label"], res["score"]
                mood = fuse_mood([], label)

                self.transcript_var.set(f"Transcript: {text}")
                self.sentiment_var.set(f"Voice: {label} ({score:.2f})")
                self.mood_var.set(f"Mood: {mood}")
                self.suggestion_var.set(f"Suggestion: {suggestion_for(mood)}")
                self.lbl_mood.config(
                    fg={"calm ☀️": "#27ae60",
                        "stressed 🌩️": "#e74c3c",
                        "uncertain 🌫️": "#f39c12"}.get(mood, "#34495e")
                )

                # --- User text in chat ---
                self.append_chat("🧑", text)

                # --- Generate AI reply ---
                new_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
                if chat_history_ids is None:
                    bot_input_ids = new_input_ids
                else:
                    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

                chat_history_ids = model.generate(
                    bot_input_ids,
                    max_length=500,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8
                )

                ai_reply = tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                    skip_special_tokens=True
                )

                self.append_chat("🤖", ai_reply)

            except sr.WaitTimeoutError:
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.append_chat("⚠️", f"Error: {e}")
                time.sleep(0.5)

    def append_chat(self, speaker, msg):
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert("end", f"{speaker}: {msg}\n")
        self.chat_log.see("end")
        self.chat_log.config(state=tk.DISABLED)

    def analyze_chat_text(self):
        try:
            # Get all text from chat log
            chat_content = self.chat_log.get("1.0", "end").strip().splitlines()
            # Find last user message
            last_user = None
            for line in reversed(chat_content):
                if line.startswith("🧑:"):
                    last_user = line.replace("🧑:", "").strip()
                    break

            if not last_user:
                messagebox.showinfo("Info", "No user message found in chat.")
                return

            # Run sentiment
            sentiment = pipeline("sentiment-analysis")
            res = sentiment(last_user)[0]
            label, score = res["label"], res["score"]

            # Update UI
            mood = fuse_mood([], label)
            self.transcript_var.set(f"Transcript (chat): {last_user}")
            self.sentiment_var.set(f"Chat: {label} ({score:.2f})")
            self.mood_var.set(f"Mood: {mood}")
            self.suggestion_var.set(f"Suggestion: {suggestion_for(mood)}")
            self.lbl_mood.config(
                fg={"calm ☀️": "#27ae60",
                    "stressed 🌩️": "#e74c3c",
                    "uncertain 🌫️": "#f39c12"}.get(mood, "#34495e")
            )

        except Exception as e:
            messagebox.showerror("Error", f"Chat mood analysis failed: {e}")


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
