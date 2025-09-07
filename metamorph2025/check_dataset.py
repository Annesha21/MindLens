import os

base = "images"  # change this if your dataset folder has another name
for sub in os.listdir(base):
    path = os.path.join(base, sub)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        print(f"{sub}: {count}")
