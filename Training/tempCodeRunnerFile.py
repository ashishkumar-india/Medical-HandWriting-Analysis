# train_hwr_ctc.py
import os, csv, math, random, string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------
# Config
# -----------------------
CSV_PATH = "training_labels.csv"          # columns: image_path,text
IMAGE_ROOT = "training_words"                        # optional prefix to join with image_path
IMG_HEIGHT = 48                        # fixed height after resize
MAX_WIDTH = 512                        # pad width per batch; can increase
BATCH_SIZE = 16
EPOCHS = 30
VAL_SPLIT = 0.1
SEED = 42

# -----------------------
# Load CSV and build charset
# -----------------------
samples = []
charset = set()
with open(CSV_PATH, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = row["image_path"]   # <-- fix here
        if IMAGE_ROOT and not os.path.isabs(p):
            p = os.path.join(IMAGE_ROOT, p)
        text = row["text"]
        if not os.path.exists(p) or text is None:
            continue
        samples.append((p, text))
        charset.update(list(text))

# Sort charset for reproducibility and make mappings
charset = sorted(list(charset))
char_to_id = {c: i for i, c in enumerate(charset)}
id_to_char = {i: c for c, i in char_to_id.items()}
vocab_size = len(charset)  # CTC uses extra "blank" internally as +1 class

# Train/val split
random.Random(SEED).shuffle(samples)
n_val = int(len(samples) * VAL_SPLIT)
val_samples = samples[:n_val]
train_samples = samples[n_val:]

def encode_label(text):
    return [char_to_id[c] for c in text]

# -----------------------
# Image loader and preprocessing
# -----------------------
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1, dtype=tf.uint8, name=None)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img

def resize_keep_aspect(img, target_h=IMG_HEIGHT):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    scale = tf.cast(target_h, tf.float32) / tf.cast(h, tf.float32)
    new_w = tf.cast(tf.math.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, (target_h, new_w), method="bilinear")
    return img

@tf.function
def augment(img):
    # Light augmentations that preserve legibility
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    # Small rotation via affine transform (approximate)
    angle = tf.random.uniform([], -0.05, 0.05)  # ~±2.8 degrees
    img = tfa.image.rotate(img, angles=angle, fill_mode="constant") if tf.executing_eagerly() else img
    return img

def prepare_example(path, label, training):
    img = load_image(path)
    img = resize_keep_aspect(img, IMG_HEIGHT)
    # Optional: augment
    if training:
        try:
            import tensorflow_addons as tfa
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tfa.image.rotate(img, tf.random.uniform([], -0.05, 0.05), fill_mode="constant")
        except ImportError:
            pass
    # Ensure shape [H, W, 1]
    img = tf.ensure_shape(img, [IMG_HEIGHT, None, 1])
    label_ids = tf.cast(tf.convert_to_tensor(encode_label(label)), tf.int32)
    label_len = tf.shape(label_ids)
    width = tf.shape(img)[8]
    return {"image": img, "label": label_ids, "label_len": label_len, "width": width}

def make_dataset(samples, training=True):
    paths = [p for p, _ in samples]
    labels = [t for _, t in samples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, t: prepare_example(p, t, training),
                num_parallel_calls=tf.data.AUTOTUNE)
    # Bucket by width to reduce padding waste
    def element_length(x):
        return x["width"]
    bucket_boundaries = [128, 192, 256, 320, 384, 448, 512, 640, 768]
    bucket_batch_sizes = [BATCH_SIZE] * (len(bucket_boundaries) + 1)
    ds = ds.apply(tf.data.experimental.bucket_by_sequence_length(
        element_length, bucket_boundaries, bucket_batch_sizes,
        padded_shapes={
            "image": [IMG_HEIGHT, None, 1],
            "label": [None],
            "label_len": [],
            "width": []
        },
        padding_values={
            "image": 0.0,
            "label": tf.cast(-1, tf.int32),  # will be masked later
            "label_len": tf.cast(0, tf.int32),
            "width": tf.cast(0, tf.int32)
        }
    ))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_samples, training=True)
val_ds = make_dataset(val_samples, training=False)

# -----------------------
# Model: CNN → BiLSTM → Dense(|V|+1)
# -----------------------
def build_model(vocab_size):
    inputs = keras.Input(shape=(IMG_HEIGHT, None, 1), name="image", dtype="float32")
    # CNN encoder with overall horizontal stride of 4
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D(pool_size=(2,2))(x)          # H/2, W/2
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)          # H/4, W/4
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    # Collapse height dimension to 1 by averaging (could also use MaxPool along H to 1)
    x = tf.reduce_mean(x, axis=1)                     # shape: [B, W/4, C]
    # Sequence model
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    logits = layers.Dense(vocab_size + 1, activation="linear")(x)  # +1 for CTC blank
    return keras.Model(inputs, logits, name="crnn_ctc")

model = build_model(vocab_size)
# Note: time steps = width // 4 after the encoder due to pooling stride
total_stride = 4

# -----------------------
# CTC loss and metrics
# -----------------------
@tf.function
def ctc_loss(y_true, y_pred, input_len, label_len):
    # y_pred: [B, T, V+1] logits
    y_pred_soft = tf.nn.log_softmax(y_pred, axis=-1)
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred_soft,
        label_length=label_len,
        logit_length=input_len,
        logits_time_major=False,
        blank_index=vocab_size
    )
    return tf.reduce_mean(loss)

def greedy_decode(logits):
    # logits: [B, T, V+1]
    pred = tf.argmax(logits, axis=-1, output_type=tf.int32)  # [B, T]
    # collapse repeats and remove blanks
    blank = vocab_size
    sequences = []
    for row in pred.numpy():
        seq = []
        prev = -1
        for p in row:
            if p == blank:
                prev = -1
                continue
            if p != prev:
                seq.append(p)
            prev = p
        sequences.append(seq)
    return sequences

def ids_to_text(ids):
    return "".join(id_to_char[i] for i in ids if i in id_to_char)

# -----------------------
# Custom training loop to handle variable lengths
# -----------------------
optimizer = keras.optimizers.Adam(1e-3)

@tf.function
def train_step(batch):
    # Compute input_length (time steps after stride) per sample
    widths = batch["width"]
    input_len = tf.cast(tf.math.floordiv(widths, total_stride), tf.int32)
    # Pad labels to max in batch and make Ragged for tf.nn.ctc_loss
    y_true = tf.RaggedTensor.from_tensor(
        tf.keras.utils.pad_sequences(batch["label"], padding="post", value=-1),
        padding=-1
    )
    label_len = batch["label_len"]
    with tf.GradientTape() as tape:
        logits = model(batch["image"], training=True)  # [B, T, V+1]
        loss = ctc_loss(y_true, logits, input_len, label_len)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_step(batch):
    widths = batch["width"]
    input_len = tf.cast(tf.math.floordiv(widths, total_stride), tf.int32)
    y_true = tf.RaggedTensor.from_tensor(
        tf.keras.utils.pad_sequences(batch["label"], padding="post", value=-1),
        padding=-1
    )
    label_len = batch["label_len"]
    logits = model(batch["image"], training=False)
    loss = ctc_loss(y_true, logits, input_len, label_len)
    return loss, logits

def cer(ref, hyp):
    # Simple Levenshtein CER
    import numpy as np
    dp = [[i+j if i*j==0 else 0 for j in range(len(hyp)+1)] for i in range(len(ref)+1)]
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1]==hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1] / max(1, len(ref))

# -----------------------
# Training loop
# -----------------------
best_val = float("inf")
for epoch in range(1, EPOCHS+1):
    # Train
    losses = []
    for batch in train_ds:
        loss = train_step(batch)
        losses.append(loss.numpy())
    train_loss = sum(losses)/len(losses)

    # Validate
    v_losses = []
    all_cer = []
    for batch in val_ds:
        v_loss, logits = val_step(batch)
        v_losses.append(v_loss.numpy())
        # Decode a few
        pred_ids = greedy_decode(logits)
        gt_texts = []
        for lab in batch["label"].numpy():
            gt_texts.append("".join(id_to_char[i] for i in lab if i != -1))
        hyp_texts = [ids_to_text(ids) for ids in pred_ids]
        # CER on batch
        for r, h in zip(gt_texts, hyp_texts):
            all_cer.append(cer(r, h))
    val_loss = sum(v_losses)/len(v_losses)
    val_cer = sum(all_cer)/max(1, len(all_cer))
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_CER={val_cer:.3f}")

    # Save best
    if val_cer < best_val:
        best_val = val_cer
        model.save("crnn_ctc_best.h5")

print("Training done. Best CER:", best_val)