import os, csv, tensorflow as tf
from tensorflow import keras

# -----------------------
# Config
# -----------------------
CSV_PATH = "validation_labels.csv"     # columns: IMAGE,MEDICINE_NAME
IMAGE_ROOT = "validation_words"        # folder where validation images are stored
IMG_HEIGHT = 48

# Load trained model
model = keras.models.load_model("my_model.keras", compile=False)

# Load charset from training (IMPORTANT: same mapping as training time)
# Make sure you saved char_to_id and id_to_char during training
import pickle
with open("charset.pkl", "rb") as f:
    char_to_id, id_to_char = pickle.load(f)
vocab_size = len(char_to_id)

# -----------------------
# Helper functions
# -----------------------
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize keeping aspect ratio
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    scale = tf.cast(IMG_HEIGHT, tf.float32) / tf.cast(h, tf.float32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    img = tf.image.resize(img, (IMG_HEIGHT, new_w))
    img = tf.expand_dims(img, 0)   # add batch dimension
    return img

def greedy_decode(logits):
    pred = tf.argmax(logits, axis=-1)[0].numpy()
    blank = vocab_size
    seq, prev = [], -1
    for p in pred:
        if p == blank:
            prev = -1
            continue
        if p != prev:
            seq.append(p)
        prev = p
    return "".join(id_to_char[i] for i in seq if i in id_to_char)

def cer(ref, hyp):
    dp = [[i+j if i*j==0 else 0 for j in range(len(hyp)+1)] for i in range(len(ref)+1)]
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1]==hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1] / max(1, len(ref))

# -----------------------
# Run validation
# -----------------------
samples = []
with open(CSV_PATH, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row["IMAGE"]
        if IMAGE_ROOT and not os.path.isabs(img_path):
            img_path = os.path.join(IMAGE_ROOT, img_path)
        if os.path.exists(img_path):
            samples.append((img_path, row["MEDICINE_NAME"]))

total_cer = []
for path, gt_text in samples:
    img = load_image(path)
    logits = model(img, training=False)
    pred_text = greedy_decode(logits)
    e = cer(gt_text, pred_text)
    total_cer.append(e)
    print(f"GT: {gt_text} | PRED: {pred_text} | CER: {e:.2f}")

print("Average CER on validation set:", sum(total_cer)/len(total_cer))
