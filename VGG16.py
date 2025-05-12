import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread
from skimage.transform import resize
from skimage.util import random_noise

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(0)

# === Load and preprocess training images from folder ===
def load_and_preprocess_images(img_dir, limit=None):
    X, Y = [], []
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if limit:
        files = files[:limit]

    for file in files:
        path = os.path.join(img_dir, file)
        img = imread(path)
        img = resize(img, (224, 224), anti_aliasing=True)

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = random_noise(img, mode='gaussian', var=0.001)

        lab = rgb2lab(img)
        L = lab[:, :, 0] / 100.0
        ab = lab[:, :, 1:] / 128.0
        X.append(L.reshape(224, 224, 1))
        Y.append(ab)
    return np.array(X), np.array(Y)

X_train, Y_train = load_and_preprocess_images("img", limit=500)

# === Pre-initialize VGG16 feature extractor for perceptual loss ===
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_model = Model(inputs=vgg_base.input, outputs=vgg_base.get_layer('block3_conv3').output)
feature_model.trainable = False

# === Perceptual Loss Helper ===
def lab2rgb_batch(lab_batch):
    rgb_batch = [lab2rgb(lab) for lab in lab_batch]
    return np.array(rgb_batch).astype(np.float32)

def perceptual_loss(y_true, y_pred):
    lab_true = tf.concat([tf.ones_like(y_true[:, :, :, :1]) * 0.5, y_true], axis=-1)
    lab_pred = tf.concat([tf.ones_like(y_pred[:, :, :, :1]) * 0.5, y_pred], axis=-1)

    rgb_true = tf.py_function(func=lab2rgb_batch, inp=[lab_true], Tout=tf.float32)
    rgb_pred = tf.py_function(func=lab2rgb_batch, inp=[lab_pred], Tout=tf.float32)

    rgb_true.set_shape([None, 224, 224, 3])
    rgb_pred.set_shape([None, 224, 224, 3])

    rgb_true = preprocess_input(rgb_true * 255.0)
    rgb_pred = preprocess_input(rgb_pred * 255.0)

    true_features = feature_model(rgb_true)
    pred_features = feature_model(rgb_pred)

    return tf.reduce_mean(tf.square(true_features - pred_features))

# === Build Colorization Model ===
def build_colorization_model():
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg.layers:
        layer.trainable = False

    input_l = Input(shape=(224, 224, 1))
    x = Conv2D(3, (1, 1), activation='linear', padding='same')(input_l)
    x = vgg(x)

    # Decoder
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)  # 7x7 -> 14x14

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)  # 14x14 -> 28x28

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 28x28 -> 56x56

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 56x56 -> 112x112

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 112x112 -> 224x224 ✅

    output_ab = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)

    model = Model(inputs=input_l, outputs=output_ab)

    # Define perceptual loss model once, outside the loss function
    perceptual_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    perceptual_layer = Model(inputs=perceptual_model.input,
                             outputs=perceptual_model.get_layer("block3_conv3").output)
    perceptual_layer.trainable = False

    def perceptual_loss(y_true, y_pred):
        def lab_to_rgb_tensor(lab_batch):
            rgb_batch = []
            for lab in lab_batch:
                rgb = lab2rgb(lab)  # skimage.color.lab2rgb
                rgb_batch.append(rgb)
            return np.array(rgb_batch).astype(np.float32)

        # Add dummy L channel (assumed 0.5)
        lab_true = tf.concat([tf.ones_like(y_true[:, :, :, :1]) * 0.5, y_true], axis=-1)
        lab_pred = tf.concat([tf.ones_like(y_pred[:, :, :, :1]) * 0.5, y_pred], axis=-1)

        # Convert LAB -> RGB using numpy_function
        rgb_true = tf.numpy_function(lab_to_rgb_tensor, [lab_true], tf.float32)
        rgb_pred = tf.numpy_function(lab_to_rgb_tensor, [lab_pred], tf.float32)

        # Manually set the shapes (very important!)
        rgb_true.set_shape([None, 224, 224, 3])
        rgb_pred.set_shape([None, 224, 224, 3])

        # Preprocess for VGG16
        rgb_true = preprocess_input(rgb_true * 255.0)
        rgb_pred = preprocess_input(rgb_pred * 255.0)

        # Get VGG features
        true_features = perceptual_layer(rgb_true)
        pred_features = perceptual_layer(rgb_pred)

        return tf.reduce_mean(tf.square(true_features - pred_features))

    def combined_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ploss = perceptual_loss(y_true, y_pred)
        return mse + 0.1 * ploss

    model.compile(optimizer=Adam(1e-4), loss=combined_loss)
    return model




model = build_colorization_model()
model.summary()

# === Training ===
checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, Y_train, batch_size=10, epochs=200, callbacks=[checkpoint, early_stop])

# === Plot Training Loss ===
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# === Load best model ===
model = load_model("best_model.h5", compile=False)

# === Predict & Save Outputs ===
def colorize_image(img_path, model, output_dir="colorized_output"):
    os.makedirs(output_dir, exist_ok=True)
    img = imread(img_path)
    img_resized = resize(img, (224, 224), anti_aliasing=True)
    lab = rgb2lab(img_resized)
    L = lab[:, :, 0] / 100.0
    L_input = L.reshape(1, 224, 224, 1)

    pred_ab = model.predict(L_input)[0] * 128
    L_orig = L * 100

    lab_out = np.zeros((224, 224, 3))
    lab_out[:, :, 0] = L_orig
    lab_out[:, :, 1:] = pred_ab
    color_rgb = lab2rgb(lab_out)

    save_path = os.path.join(output_dir, f"colorized_{os.path.basename(img_path)}")
    cv2.imwrite(save_path, cv2.cvtColor((color_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Saved: {save_path}")

# === Apply to Test Images ===
for file in os.listdir("img"):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        colorize_image(os.path.join("img", file), model)
