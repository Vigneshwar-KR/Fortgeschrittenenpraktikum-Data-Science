import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
from PIL import Image

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

model_path = "/work/i1000848/Hybrid Models/Video based models/1/best models/run18/best_model_06_feb.h5"
custom_objects = {"dice_loss": dice_loss, "MeanIoU": MeanIoU}
model = load_model(model_path, custom_objects=custom_objects, compile=False)

# # Define class mappings
# color2index = {
#     (255, 255, 255): 0,  # Background
#     (255, 0, 0): 1,      # 0° Fiber
#     (0, 0, 255): 2,      # 90° Fiber
#     (255, 255, 0): 3     # Other
# }
# index2color = {v: k for k, v in color2index.items()}  # Reverse mapping

color2index = {
    (255, 0, 0): 0,      # 0° Fiber
    (0, 0, 255): 1,      # 90° Fiber
    (255, 255, 255): 2,  # Resin
    (255, 255, 0): 3     # Other
}
index2color = {v: k for k, v in color2index.items()}  # Reverse mapping


def preprocess_image(image_path, size=512):
    try:
        img = Image.open(image_path).resize((size, size))
        img = np.asarray(img).astype(np.float32) / 255.0  # Normalize
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_gt(gt_path, size=512):
    try:
        gt = Image.open(gt_path).resize((size, size))
        gt = np.asarray(gt)

        gt_idx = np.zeros((size, size), dtype=np.uint8)
        for color, class_idx in color2index.items():
            mask = np.all(gt == np.array(color), axis=-1)
            gt_idx[mask] = class_idx
        return gt_idx
    except Exception as e:
        print(f"Error loading ground truth {gt_path}: {e}")
        return None

def save_results(rgb, gt, pred, output_path, file_name):
    os.makedirs(output_path, exist_ok=True)

    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    for idx, color in index2color.items():
        gt_rgb[gt == idx] = color
        pred_rgb[pred == idx] = color

    Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(output_path, f"{file_name}_rgb.png"))
    Image.fromarray(gt_rgb).save(os.path.join(output_path, f"{file_name}_gt.png"))
    Image.fromarray(pred_rgb).save(os.path.join(output_path, f"{file_name}_pred.png"))

def predict_multiple_images(image_folder, gt_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".png")])

    assert len(image_files) == len(gt_files), "Mismatch between number of images and ground truth masks!"

    total_accuracy = 0
    class_intersections = {i: 0 for i in range(len(index2color))}
    class_unions = {i: 0 for i in range(len(index2color))}
    count = len(image_files)

    mean_iou_metric = MeanIoU(num_classes=len(index2color))

    for img_file, gt_file in zip(image_files, gt_files):
        image_path = os.path.join(image_folder, img_file)
        gt_path = os.path.join(gt_folder, gt_file)

        rgb_img = preprocess_image(image_path)
        gt_img = preprocess_gt(gt_path)

        if rgb_img is None or gt_img is None:
            print(f"Skipping {img_file} due to loading error.")
            continue

        img_input = np.expand_dims(rgb_img, axis=0)
        pred_mask = model.predict(img_input, verbose=0)[0]
        pred_mask = np.argmax(pred_mask, axis=-1)

        file_name = os.path.splitext(img_file)[0]
        save_results(rgb_img, gt_img, pred_mask, output_folder, file_name)

        accuracy = np.mean(gt_img == pred_mask)
        total_accuracy += accuracy

        mean_iou_metric.update_state(gt_img.flatten(), pred_mask.flatten())

        for i in range(len(index2color)):
            intersection = np.logical_and(gt_img == i, pred_mask == i).sum()
            union = np.logical_or(gt_img == i, pred_mask == i).sum()
            class_intersections[i] += intersection
            class_unions[i] += union

    test_accuracy = total_accuracy / count
    test_miou = mean_iou_metric.result().numpy()

    final_class_ious = {idx: (class_intersections[idx] / class_unions[idx]) if class_unions[idx] > 0 else 0 
                        for idx in range(len(index2color))}

    print("\n--- Test Evaluation Results ---")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test mIoU: {test_miou:.4f}")
    print("\nClass-wise IoU Scores:")
    for idx, iou in final_class_ious.items():
        class_name = f"Class {idx} ({index2color[idx]})"
        print(f"  {class_name}: {iou:.4f}")

    return test_accuracy, test_miou, final_class_ious

image_folder = "/beegfs/work/fpds02/final_project_2/CFRP_dataset/CFRP_dataset/patches/augmented/train"
gt_folder = "/beegfs/work/fpds02/final_project_2/CFRP_dataset/CFRP_dataset/patches/augmented/label"
output_folder = "/beegfs/work/fpds02/final_project_2/CFRP_dataset/CFRP_dataset/patches/augmented/folder_prediction_keras_9324"

test_accuracy, test_miou, class_ious = predict_multiple_images(image_folder, gt_folder, output_folder)




