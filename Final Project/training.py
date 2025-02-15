import os
import numpy as np
from glob import glob
from natsort import natsorted
from PIL import Image
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Add
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import wandb
from wandb.integration.keras import WandbMetricsLogger


wandb.login(key="enter WandB key here ")  
wandb.init(project="CFRP_Segmentation Final", name="U-Net++_SA_Training_06_feb", config={"architecture": "UNet++", "epochs": 200,"loss": "dice loss", "type" : "cross_entropy_2", "augmentation": True})

BLACK_PIXEL_DIR = "/beegfs/work/fpds02/final_project_2/CFRP_dataset/CFRP_dataset/patches/black_pixel_removal"
AUGMENTED_DIR = "/beegfs/work/fpds02/final_project_2/CFRP_dataset/CFRP_dataset/patches/augmented"

# PLEASE CHANGE THE RUN FOLDER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODEL_SAVE_PATH = "Enter path to save the model"

IMG_SIZE = 512
BATCH_SIZE = 4
N_CLASSES = 4
EPOCHS = 200

# Old Color Map for Label Encoding
# COLOR_MAP = {
#     (255, 255, 255): 0,  # Background
#     (255, 0, 0): 1,      # 0° Fiber
#     (0, 0, 255): 2,      # 90° Fiber
#     (255, 255, 0): 3     # Other
# }

COLOR_MAP = {
    (255, 0, 0): 0,      # 0° Fiber
    (0, 0, 255): 1,      # 90° Fiber
    (255, 255, 255): 2,  # Resin
    (255, 255, 0): 3     # Other
}
INDEX_TO_COLOR = {v: k for k, v in COLOR_MAP.items()}

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def conv_block(x, filters, kernel_size=(3, 3), activation='relu', batch_norm=True):
    """Convolutional block with optional batch normalization"""
    x = Conv2D(filters, kernel_size, padding="same")(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def unet_plus_plus(img_size, n_classes):
    inputs = Input((img_size, img_size, 3))

    # Encoder
    c1 = conv_block(inputs, 64)
    c1 = conv_block(c1, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    c2 = conv_block(c2, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    c3 = conv_block(c3, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    c4 = conv_block(c4, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 1024)
    c5 = conv_block(c5, 1024)

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = Add()([u6, c4])
    c6 = conv_block(u6, 512)
    c6 = conv_block(c6, 512)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = Add()([u7, c3])
    c7 = conv_block(u7, 256)
    c7 = conv_block(c7, 256)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = Add()([u8, c2])
    c8 = conv_block(u8, 128)
    c8 = conv_block(c8, 128)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = Add()([u9, c1])
    c9 = conv_block(u9, 64)
    c9 = conv_block(c9, 64)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def load_data(black_pixel_dir, augmented_dir):
    train_files = natsorted(glob(os.path.join(black_pixel_dir, "train", "*.png")))
    label_files = natsorted(glob(os.path.join(black_pixel_dir, "label", "*.png")))

    train_aug_files = natsorted(glob(os.path.join(augmented_dir, "train", "*.png")))
    label_aug_files = natsorted(glob(os.path.join(augmented_dir, "label", "*.png")))

    train_files.extend(train_aug_files)
    label_files.extend(label_aug_files)

    assert len(train_files) == len(label_files), "Mismatch in number of train and label files!"
    return train_files, label_files

def split_data(train_files, label_files, test_size=0.25):
    return train_test_split(train_files, label_files, test_size=test_size, random_state=42)

class DataGenerator(Sequence):
    def __init__(self, image_paths, label_paths, img_size, batch_size):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.label_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        labels = []

        for img_path, label_path in zip(batch_images, batch_labels):
            img = Image.open(img_path).resize((self.img_size, self.img_size))
            label = Image.open(label_path).resize((self.img_size, self.img_size))

            img = np.array(img) / 255.0
            label = self.convert_label(label)

            images.append(img)
            labels.append(label)

        return np.array(images), np.array(labels)

    def convert_label(self, label):
        label = np.array(label)

        if len(label.shape) == 2:  # (H, W)
            h, w = label.shape
        else: 
            h, w, _ = label.shape

        label_idx = np.zeros((h, w), dtype=np.uint8)

        for color, class_idx in COLOR_MAP.items():
            mask = np.all(label == np.array(color), axis=-1) if len(label.shape) == 3 else (label == color[0])
            label_idx[mask] = class_idx

        return tf.keras.utils.to_categorical(label_idx, num_classes=N_CLASSES)


train_files, label_files = load_data(BLACK_PIXEL_DIR, AUGMENTED_DIR)
train_files, val_files, train_labels, val_labels = split_data(train_files, label_files)

train_gen = DataGenerator(train_files, train_labels, IMG_SIZE, BATCH_SIZE)
val_gen = DataGenerator(val_files, val_labels, IMG_SIZE, BATCH_SIZE)

model = unet_plus_plus(IMG_SIZE, N_CLASSES)

# model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy", MeanIoU(num_classes=N_CLASSES)])
model.compile(optimizer=Adam(learning_rate=0.001), loss=dice_loss, metrics=["accuracy", MeanIoU(num_classes=N_CLASSES)])


callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True),
    WandbMetricsLogger()
]

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
model.save(MODEL_SAVE_PATH)

print("Evaluation Results:")
print("Train:")
scores_train = model.evaluate(train_gen)
print("Validation:")
scores_val = model.evaluate(val_gen)

print(f"Train Accuracy: {round(scores_train[1] * 100, 2)}%")
print(f"Validation Accuracy: {round(scores_val[1] * 100, 2)}%")

print(f"Train mIoU score: {round(scores_train[2] * 100, 2)}")
print(f"Validation mIoU score: {round(scores_val[2] * 100, 2)}")

