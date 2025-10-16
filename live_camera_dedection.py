import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
from datetime import datetime
from tkinter import Tk, filedialog, messagebox, Button, Label

# --------------------------
# Paths
# --------------------------
train_csv = "Training/training_labels.csv"
model_path = "doctor_handwriting_model.keras"
output_dir = "captured_images"
csv_log_path = "captured_log.csv"

# Create output folder if not exists
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Load CSV & Generate Class Labels
# --------------------------
train_df = pd.read_csv(train_csv)
class_labels = sorted(train_df['MEDICINE_NAME'].unique())
print("Class Labels Loaded:", len(class_labels))

# --------------------------
# Load Trained Model
# --------------------------
model = load_model(model_path)
print("✅ Model loaded successfully!")

# --------------------------
# Prepare CSV logging
# --------------------------
if os.path.exists(csv_log_path):
    log_df = pd.read_csv(csv_log_path)
else:
    log_df = pd.DataFrame(columns=["timestamp","image_file","top1","top2","top3"])

# --------------------------
# Prediction Function
# --------------------------
def predict_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))
    img_array = np.expand_dims(img / 255.0, axis=0)

    pred = model.predict(img_array, verbose=0)
    top_indices = pred[0].argsort()[-3:][::-1]
    top_labels = [class_labels[i] for i in top_indices]
    top_scores = [pred[0][i] for i in top_indices]

    # Save captured image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_file = f"{output_dir}/captured_{timestamp}_{top_labels[0]}.jpg"
    cv2.imwrite(image_file, frame)

    # Log to CSV
    global log_df
    log_df = pd.concat([log_df, pd.DataFrame([{
        "timestamp": timestamp,
        "image_file": image_file,
        "top1": f"{top_labels[0]} ({top_scores[0]:.2f})",
        "top2": f"{top_labels[1]} ({top_scores[1]:.2f})",
        "top3": f"{top_labels[2]} ({top_scores[2]:.2f})"
    }])], ignore_index=True)
    log_df.to_csv(csv_log_path, index=False)

    return top_labels, top_scores

# --------------------------
# Show Result in GUI
# --------------------------
def show_result(top_labels, top_scores):
    messagebox.showinfo(
        "Prediction Result",
        f"Top 1: {top_labels[0]} ({top_scores[0]:.2f})\n"
        f"Top 2: {top_labels[1]} ({top_scores[1]:.2f})\n"
        f"Top 3: {top_labels[2]} ({top_scores[2]:.2f})"
    )

# --------------------------
# Select Image
# --------------------------
def select_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files","*.jpg *.jpeg *.png")]
    )
    if file_path:
        img = cv2.imread(file_path)
        top_labels, top_scores = predict_image(img)
        show_result(top_labels, top_scores)

# --------------------------
# Select Video
# --------------------------
def select_video():
    file_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video files","*.mp4 *.avi")]
    )
    if file_path:
        cap_file = cv2.VideoCapture(file_path)
        last_labels, last_scores = None, None

        while True:
            ret_file, frame_file = cap_file.read()
            if not ret_file:
                break
            last_labels, last_scores = predict_image(frame_file)

        cap_file.release()
        if last_labels:
            show_result(last_labels, last_scores)

# --------------------------
# GUI Main Window
# --------------------------
def main_gui():
    root = Tk()
    root.title("Doctor Handwriting Recognition")
    root.geometry("400x250")

    Label(root, text="Medicine Prediction System", font=("Arial", 16, "bold")).pack(pady=20)

    Button(root, text="Select Image", font=("Arial", 12), width=20, command=select_image).pack(pady=10)
    Button(root, text="Select Video", font=("Arial", 12), width=20, command=select_video).pack(pady=10)
    Button(root, text="Exit", font=("Arial", 12), width=20, command=root.quit).pack(pady=10)

    root.mainloop()

# --------------------------
# Run GUI
# --------------------------
if __name__ == "__main__":
    main_gui()
