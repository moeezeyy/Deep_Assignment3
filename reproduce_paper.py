# CURSOR PROMPT: Reproduce the paper "Flower Pictures Recognition Based on the Advanced Convolutional Neural Network with Oxford Flowers 102 Dataset"

# Paper (for reference): 133308.pdf

# This script reproduces the model & experiments described in the paper. It is self-contained and includes installation steps.

# Requirements: Python 3.8+, pip. Recommended: run in a virtualenv or conda env with GPU.



# ---------- STEP 0: Install dependencies (the script will attempt pip install if needed) ----------

import sys, subprocess, pkgutil, os, random

required = ["tensorflow", "matplotlib", "numpy", "pandas"]

for pkg in required:

    if not pkgutil.find_loader(pkg):

        print(f"Installing {pkg} ...")

        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])



# ---------- STEP 1: Imports and reproducibility ----------

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import datetime

from pathlib import Path



# Reproducible seeds

SEED = 42

tf.random.set_seed(SEED)

np.random.seed(SEED)

random.seed(SEED)

os.environ['PYTHONHASHSEED'] = str(SEED)



print("TensorFlow version:", tf.__version__)



# ---------- STEP 2: Configs (matches paper) ----------

IMG_SIZE = 200                   # resize to 200x200 as in the paper

BATCH_SIZE = 32

NUM_CLASSES = 102

EPOCHS = 100                      # paper used 100 epochs

AUTOTUNE = tf.data.AUTOTUNE

OUTPUT_DIR = Path("flower_repro_output")

OUTPUT_DIR.mkdir(exist_ok=True)



# ---------- STEP 3: Data loading from local dataset ----------

# Load from local directories: dataset/train, dataset/valid, dataset/test

DATASET_DIR = Path("dataset")

train_dir = DATASET_DIR / "train"

val_dir = DATASET_DIR / "valid"

test_dir = DATASET_DIR / "test"



# Check if local dataset exists

if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():

    raise RuntimeError(f"Local dataset not found. Expected directories: {train_dir}, {val_dir}, {test_dir}")



print("Loading dataset from local directories...")

print(f"Train directory: {train_dir}")

print(f"Validation directory: {val_dir}")

print(f"Test directory: {test_dir}")



# Get sorted class directories (numerically sorted, not lexicographically)

def get_sorted_classes(directory):

    """Get class directories sorted numerically (1, 2, ..., 102)"""

    class_dirs = [d.name for d in directory.iterdir() if d.is_dir()]

    # Sort numerically by converting to int

    class_dirs_sorted = sorted(class_dirs, key=lambda x: int(x))

    return class_dirs_sorted



# Create label mapping: directory name -> 0-indexed label

# e.g., "1" -> 0, "2" -> 1, ..., "102" -> 101

train_classes = get_sorted_classes(train_dir)

class_to_idx = {class_name: idx for idx, class_name in enumerate(train_classes)}

print(f"Found {len(train_classes)} classes. Label mapping: {dict(list(class_to_idx.items())[:5])}... (showing first 5)")



# Load datasets using image_dataset_from_directory with custom class names

# We'll use label_mode='int' but need to ensure proper mapping

# Since directories are already sorted numerically by our function, we can use the default behavior

# But to be safe, we'll load and manually remap labels

train_ds_raw = tf.keras.utils.image_dataset_from_directory(

    train_dir,

    label_mode='int',

    batch_size=None,

    image_size=(IMG_SIZE, IMG_SIZE),

    shuffle=True,

    seed=SEED,

    class_names=train_classes  # Ensure consistent ordering

)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(

    val_dir,

    label_mode='int',

    batch_size=None,

    image_size=(IMG_SIZE, IMG_SIZE),

    shuffle=False,

    seed=SEED,

    class_names=train_classes  # Use same class order

)

# Test directory has images directly in it (not in subdirectories), so we need to load differently
# Check if test directory has subdirectories or images directly
test_has_subdirs = any(d.is_dir() for d in test_dir.iterdir())

if test_has_subdirs:
    # Test set has class subdirectories (same structure as train/val)
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode='int',
        batch_size=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False,
        seed=SEED,
        class_names=train_classes  # Use same class order
    )
else:
    # Test set has images directly in the directory (no labels available)
    # Load images without labels for prediction only
    print("Test directory has images directly (no class subdirectories). Loading without labels...")
    test_image_paths = list(test_dir.glob("*.jpg"))
    test_count_files = len(test_image_paths)
    print(f"Found {test_count_files} test images")
    
    # Create a dataset from file paths (without labels)
    def load_image(path):
        image = tf.io.read_file(str(path))
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        # Don't normalize here - preprocess_image will do it
        image = tf.cast(image, tf.float32)
        return image
    
    # Create dataset from file paths
    test_paths_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in test_image_paths])
    test_ds_raw = test_paths_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    # Add dummy labels (0) for compatibility with evaluation - these won't be used for actual evaluation
    test_ds_raw = test_ds_raw.map(lambda img: (img, tf.constant(0, dtype=tf.int32)))

# The labels are already 0-indexed correctly since we provided class_names in sorted order

train_ds = train_ds_raw

val_ds = val_ds_raw

test_ds = test_ds_raw

# Count examples

train_count = len(list(train_dir.glob("*/*.jpg")))

val_count = len(list(val_dir.glob("*/*.jpg")))

# Test count depends on structure
if test_has_subdirs:
    test_count = len(list(test_dir.glob("*/*.jpg")))
else:
    test_count = len(list(test_dir.glob("*.jpg")))

print(f"Train examples: {train_count}, Validation: {val_count}, Test: {test_count}")



# ---------- STEP 4: Preprocessing & augmentations (as stated in paper) ----------

def preprocess_image(image, label, training=False):

    # Convert to float32 (images from image_dataset_from_directory are uint8)

    image = tf.cast(image, tf.float32)

    # Note: image_dataset_from_directory already resized to IMG_SIZE x IMG_SIZE

    # Normalize to [0,1] (images are in [0, 255] range)

    image = image / 255.0

    if training:

        # Random horizontal flip

        image = tf.image.random_flip_left_right(image, seed=SEED)

        # Random vertical flip (paper mentions up-down flipping)

        # Implement vertical flip randomly with 50% prob

        if tf.random.uniform([], seed=SEED) > 0.5:

            image = tf.image.flip_up_down(image)

        # Random brightness

        image = tf.image.random_brightness(image, 0.2, seed=SEED)

        # Random contrast

        image = tf.image.random_contrast(image, 0.8, 1.2, seed=SEED)

        # Random saturation

        image = tf.image.random_saturation(image, 0.8, 1.2, seed=SEED)

        # Random hue

        image = tf.image.random_hue(image, 0.05, seed=SEED)

        # Ensure pixel range is still clipped

        image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label



def prepare(ds, training=False):

    if training:

        ds = ds.shuffle(2048, seed=SEED)

    ds = ds.map(lambda x,y: preprocess_image(x,y, training=training), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(AUTOTUNE)

    return ds



train_ds = prepare(train_ds, training=True)

val_ds = prepare(val_ds, training=False)

test_ds = prepare(test_ds, training=False)



# ---------- STEP 5: Build the model (architecture exactly as the paper describes) ----------

from tensorflow.keras import layers, regularizers, models



def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):

    l2 = 1e-4  # small L2 regularization (paper said L2; using a typical small value)

    model = models.Sequential([

        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),

        layers.MaxPooling2D((2,2)),



        layers.Conv2D(64, (3,3), activation='relu', padding='same'),

        layers.MaxPooling2D((2,2)),



        layers.Conv2D(128, (3,3), activation='relu', padding='same'),

        layers.Conv2D(256, (3,3), activation='relu', padding='same'),

        layers.MaxPooling2D((2,2)),



        layers.GlobalAveragePooling2D(),



        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2)),

        layers.Dropout(0.5, seed=SEED),



        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2)),

        layers.Dropout(0.5, seed=SEED),



        layers.Dense(num_classes, activation='softmax')

    ])

    return model



model = build_model()

model.summary()



# ---------- STEP 6: Compile (matches paper: Adam) ----------

model.compile(

    optimizer=tf.keras.optimizers.Adam(),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=['accuracy']

)



# ---------- STEP 7: Callbacks ----------

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint_path = OUTPUT_DIR / f"best_model_{timestamp}.h5"

history_csv = OUTPUT_DIR / f"history_{timestamp}.csv"



callbacks = [

    tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', save_best_only=True, verbose=1),

    tf.keras.callbacks.CSVLogger(str(history_csv)),

    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),

    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

]



# ---------- STEP 8: Train ----------

try:

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        callbacks=callbacks

    )

except Exception as e:

    print("Training failed. Error:")

    raise



# Save final model (in case ModelCheckpoint didn't)

final_model_path = OUTPUT_DIR / f"final_model_{timestamp}.h5"

model.save(final_model_path)

print("Saved final model to", final_model_path)



# ---------- STEP 9: Evaluate on test set ----------

# Check again if test set has subdirectories (for evaluation)
test_has_subdirs_eval = any(d.is_dir() for d in test_dir.iterdir())

# Note: If test set doesn't have true labels, we'll skip test evaluation
# and use validation set as the final evaluation metric
if test_has_subdirs_eval:
    test_results = model.evaluate(test_ds, verbose=1)
    print("Test loss, Test accuracy:", test_results)
else:
    print("Test set doesn't have labels. Skipping test evaluation.")
    print("Using validation set as final evaluation metric.")
    val_results = model.evaluate(val_ds, verbose=1)
    print("Validation loss, Validation accuracy (final):", val_results)



# ---------- STEP 10: Save history, plot accuracy & loss (visual analysis like paper) ----------

# Save history as CSV (CSVLogger already wrote stepwise, but we save again for completeness)

hist_df = pd.DataFrame(history.history)

hist_csv_path = OUTPUT_DIR / f"history_dataframe_{timestamp}.csv"

hist_df.to_csv(hist_csv_path, index=False)

print("Saved history CSV to", hist_csv_path)



# Plot training & validation accuracy and loss

plt.figure(figsize=(8,4))

plt.plot(history.history['accuracy'], label='train_acc')

plt.plot(history.history['val_accuracy'], label='val_acc')

plt.title('Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

acc_plot = OUTPUT_DIR / f"accuracy_{timestamp}.png"

plt.savefig(acc_plot)

plt.close()

print("Saved accuracy plot to", acc_plot)



plt.figure(figsize=(8,4))

plt.plot(history.history['loss'], label='train_loss')

plt.plot(history.history['val_loss'], label='val_loss')

plt.title('Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

loss_plot = OUTPUT_DIR / f"loss_{timestamp}.png"

plt.savefig(loss_plot)

plt.close()

print("Saved loss plot to", loss_plot)



# ---------- STEP 11: Show some predictions & feature-map-like visualizations (as in paper) ----------

import itertools

# Take a batch from test set

for images, labels in test_ds.take(1):

    preds = model.predict(images)

    preds_top = np.argmax(preds, axis=1)

    # Show first 8 images with predicted vs true

    n = min(8, images.shape[0])

    plt.figure(figsize=(16,6))

    for i in range(n):

        ax = plt.subplot(2, 4, i+1)

        plt.imshow(images[i].numpy())

        plt.title(f"True: {int(labels[i].numpy())} | Pred: {int(preds_top[i])}")

        plt.axis('off')

    preds_plot = OUTPUT_DIR / f"predictions_{timestamp}.png"

    plt.savefig(preds_plot)

    plt.close()

    print("Saved sample predictions to", preds_plot)

    break



# Optional: visualize early conv filters & feature maps for one image (simple approach)

layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

if layer_outputs:

    # Build a small model to fetch conv outputs

    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs[:3])  # first 3 conv layers

    sample_img = images[0:1]

    activations = activation_model.predict(sample_img)

    # Save visualization for first conv activation maps

    for idx, act in enumerate(activations):

        # act shape: (1, h, w, channels)

        n_channels = act.shape[-1]

        size = act.shape[1]

        # Display only first 8 channels for brevity

        n_display = min(8, n_channels)

        plt.figure(figsize=(12,3))

        for ch in range(n_display):

            ax = plt.subplot(1, n_display, ch+1)

            plt.imshow(act[0, :, :, ch], cmap='viridis')

            plt.axis('off')

        fmap_path = OUTPUT_DIR / f"featuremap_layer{idx+1}_{timestamp}.png"

        plt.savefig(fmap_path)

        plt.close()

        print("Saved feature maps to", fmap_path)

else:

    print("No Conv2D layers found for visualization.")



# ---------- STEP 12: Final notes and outputs ----------

print("All outputs saved in folder:", OUTPUT_DIR.resolve())

print("Model checkpoint:", checkpoint_path)

print("Final model:", final_model_path)

print("Training history CSV:", hist_csv_path)

print("Paper used for reference (local path): 133308.pdf")



# End of script

