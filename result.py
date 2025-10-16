import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import os

# ----------------------------
# Paths (make sure CSV exists)
# ----------------------------
train_csv = "Training/training_labels.csv"
train_dir = "Training/training_words"

# Load class labels from training CSV
train_df = pd.read_csv(train_csv)
class_labels = sorted(train_df["MEDICINE_NAME"].unique())

# ----------------------------
# Load trained model
# ----------------------------
model = load_model("doctor_handwriting_model.keras")

# ----------------------------
# GUI Window
# ----------------------------
root = tk.Tk()
root.title("Medicine Name Prediction")
root.geometry("600x600")
root.config(bg="white")

# Label for prediction text
prediction_label = Label(
    root,
    text="Upload an image to predict medicine",
    font=("Arial", 14),
    bg="white",
    wraplength=500,
)
prediction_label.pack(pady=20)

# Label for showing uploaded image
img_label = Label(root, bg="white")
img_label.pack(pady=10)

# ----------------------------
# Prediction Function
# ----------------------------
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    
    if not file_path:
        return

    # Load image and force RGB
    img = Image.open(file_path).convert("RGB")
    img_resized = img.resize((128, 128))

    # Show in GUI
    tk_img = ImageTk.PhotoImage(img_resized)
    img_label.config(image=tk_img)
    img_label.image = tk_img

    # Preprocess for model
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred) * 100

    # Debug print
    print("Prediction:", class_labels[predicted_class], "Confidence:", confidence)

    # Show result in GUI
    prediction_label.config(
        text=f"Predicted Medicine: {class_labels[predicted_class]}\nConfidence: {confidence:.2f}%"
    )
    root.update_idletasks()

# ----------------------------
# Upload Button
# ----------------------------
upload_btn = Button(
    root,
    text="Upload Image",
    command=upload_and_predict,
    font=("Arial", 12),
    bg="blue",
    fg="white",
    padx=10,
    pady=5,
)
upload_btn.pack(pady=20)

# Run GUI
root.mainloop()
