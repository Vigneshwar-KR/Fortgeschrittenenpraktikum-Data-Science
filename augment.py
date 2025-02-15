import os
import shutil
from glob import glob
from natsort import natsorted
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.core.composition import OneOf
import random
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"




def augment_and_save(train_dir, label_dir, output_dir, augmentations, num_augmentations=3):
    """
    Augment train and label images and save them in a new directory structure.

    Args:
        train_dir (str): Path to the directory containing training images.
        label_dir (str): Path to the directory containing label images.
        output_dir (str): Path to save the augmented data.
        augmentations (A.Compose): Albumentations augmentation pipeline.
        num_augmentations (int): Number of augmented samples to create per image.

    Returns:
        None
    """
    # Ensure output directory exists
    train_output_dir = os.path.join(output_dir, "train")
    label_output_dir = os.path.join(output_dir, "label")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # Get train and label files
    train_files = natsorted(glob(os.path.join(train_dir, "*.png")))
    label_files = natsorted(glob(os.path.join(label_dir, "*.png")))

    # Check if every train image has a corresponding label
    assert len(train_files) == len(label_files), "Mismatch in number of train and label files!"

    for train_file, label_file in zip(train_files, label_files):
        assert os.path.basename(train_file) == os.path.basename(label_file), f"File mismatch: {train_file} and {label_file}"

    print(f"Total number of images before augmentation: {len(train_files)}")

    # Perform augmentations
    for idx, (train_file, label_file) in enumerate(zip(train_files, label_files)):
        train_img = np.array(Image.open(train_file))
        label_img = np.array(Image.open(label_file))

        for aug_idx in range(num_augmentations):
            augmented = augmentations(image=train_img, mask=label_img)

            # Save augmented images
            train_aug_path = os.path.join(train_output_dir, f"{idx}_{aug_idx}.png")
            label_aug_path = os.path.join(label_output_dir, f"{idx}_{aug_idx}.png")

            Image.fromarray(augmented['image']).save(train_aug_path)
            Image.fromarray(augmented['mask']).save(label_aug_path)

    # Calculate total number of images after augmentation
    total_augmented_train = len(glob(os.path.join(train_output_dir, "*.png")))
    total_augmented_label = len(glob(os.path.join(label_output_dir, "*.png")))

    print(f"Total number of images after augmentation (train): {total_augmented_train}")
    print(f"Total number of images after augmentation (label): {total_augmented_label}")

    # Calculate total number of images before and after augmentation
    total_before = len(train_files)
    total_after = total_augmented_train

    print(f"Total number of images before augmentation: {total_before}")
    print(f"Total number of images (before + augmented): {total_before + total_after}")

if __name__ == "__main__":
    TRAIN_DIR = "DLR_Project/FPDS_Project/CFRP_dataset/black_pixel_removal/train/"
    LABEL_DIR = "DLR_Project/FPDS_Project/CFRP_dataset/black_pixel_removal/label/"
    OUTPUT_DIR = "DLR_Project/FPDS_Project/CFRP_dataset/augmented/"


    # Define augmentation pipeline with only scale for zoom-in
    augmentations = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5)
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(p=0.2),
            A.GaussNoise(p=0.2)
        ], p=0.3),
        A.Affine(scale=(1.1, 1.3), p=0.5)  # Scale only for zoom-in
    ])

    augment_and_save(TRAIN_DIR, LABEL_DIR, OUTPUT_DIR, augmentations, num_augmentations=3)

