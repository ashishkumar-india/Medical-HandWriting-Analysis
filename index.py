import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# --------------------------
# Paths
# --------------------------
train_dir = "Training/training_words"
val_dir   = "Validation/validation_words"
test_dir  = "Testing/testing_words"

train_csv = "Training/training_labels.csv"
val_csv   = "Validation/validation_labels.csv"
test_csv  = "Testing/testing_labels.csv"

# --------------------------
# Load CSVs
# --------------------------
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
test_df  = pd.read_csv(test_csv)

# --------------------------
# Image Data Generators
# --------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# --------------------------
# Generators
# --------------------------
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="IMAGE",
    y_col="MEDICINE_NAME",
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=val_dir,
    x_col="IMAGE",
    y_col="MEDICINE_NAME",
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="IMAGE",
    y_col="MEDICINE_NAME",
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# --------------------------
# Build CNN Model
# --------------------------
num_classes = len(train_generator.class_indices)

model = Sequential([
    Input(shape=(128,128,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------------
# Train Model
# --------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)

# --------------------------
# Save Model
# --------------------------
model.save("doctor_handwriting_model.keras")
print("✅ Model saved successfully!")

# --------------------------
# Evaluate on Testing Set
# --------------------------
loss, accuracy = model.evaluate(test_generator)
print("Test Accuracy:", accuracy)

# --------------------------
# Real-Time Camera Prediction
# --------------------------
cap = cv2.VideoCapture(0)
class_labels = list(train_generator.class_indices.keys())

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    img = cv2.resize(frame, (128,128))
    img_array = np.expand_dims(img / 255.0, axis=0)

    pred = model.predict(img_array)
    predicted_class = np.argmax(pred, axis=1)
    label = class_labels[predicted_class[0]]

    # Display predicted label on frame
    cv2.putText(frame, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Doctor Handwriting Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
