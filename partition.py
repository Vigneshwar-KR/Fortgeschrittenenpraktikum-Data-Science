import cv2
import os

# def divide_image(image_path, output_folder):
#     """
#     Divide a high-resolution image into 8 equal parts and save them.
    
#     Args:
#         image_path (str): Path to the input high-resolution image.
#         output_folder (str): Folder to save the divided image parts.
#     """
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Image not found or failed to load.")
#         return

#     # Get image dimensions
#     height, width, _ = image.shape

#     # Calculate dimensions for 8 equal parts
#     part_width = width // 4
#     part_height = height // 2

#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Split the image into 8 parts
#     part_count = 0
#     for i in range(2):  # 2 rows
#         for j in range(4):  # 4 columns
#             # Define the coordinates of the part
#             x_start = j * part_width
#             y_start = i * part_height
#             x_end = x_start + part_width
#             y_end = y_start + part_height

#             # Crop the image part
#             part = image[y_start:y_end, x_start:x_end]

#             # Save the image part
#             part_path = os.path.join(output_folder, f"part_{part_count}.jpg")
#             cv2.imwrite(part_path, part)
#             print(f"Saved: {part_path}")
#             part_count += 1

#     print(f"Image divided into 8 parts and saved in '{output_folder}'.")

# # Example usage
# image_path = "train_img_3.jpg"  # Replace with your image file path
# output_folder = "Train image 3"  # Folder to save the divided parts
# divide_image(image_path, output_folder)



#############################################################################


import cv2
import numpy as np
import os

def combine_images(input_folder, output_file, original_shape):
    """
    Combine 8 equal parts of an image back into the original high-resolution image.

    Args:
        input_folder (str): Folder containing the 8 parts.
        output_file (str): Path to save the combined image.
        original_shape (tuple): The original shape of the high-resolution image (height, width, channels).
    """
    # Dimensions of the original image
    original_height, original_width, channels = original_shape

    # Calculate dimensions of each part
    part_width = original_width // 4
    part_height = original_height // 2

    # Initialize an empty image for combining
    combined_image = np.zeros((original_height, original_width, channels), dtype=np.uint8)

    # Load the parts and combine them
    part_count = 0
    for i in range(2):  # 2 rows
        for j in range(4):  # 4 columns
            # Construct the file name
            part_file = os.path.join(input_folder, f"part_{part_count}_label.tif")

            # Load the image part
            part = cv2.imread(part_file)
            if part is None:
                print(f"Error: Failed to load {part_file}. Skipping.")
                continue

            # Validate part size
            if part.shape[:2] != (part_height, part_width):
                raise ValueError(f"Part {part_file} has incorrect dimensions: {part.shape[:2]}")

            # Define the coordinates of where to place the part in the combined image
            x_start = j * part_width
            y_start = i * part_height
            x_end = x_start + part_width
            y_end = y_start + part_height

            # Place the part in the combined image
            combined_image[y_start:y_end, x_start:x_end] = part
            part_count += 1

    # Save the combined image
    cv2.imwrite(output_file, combined_image)
    print(f"Combined image saved as: {output_file}")


# Example usage
input_folder = r"C:\Vicky\Tu Braunschweig\Semester 5\FPDS\DLR_Project\Train image 1\label"  # Folder containing the 8 parts
output_file = r"C:\Vicky\Tu Braunschweig\Semester 5\FPDS\DLR_Project\Train image 1\combined_image_label.tif"  # Output file name
original_shape = (4613, 17573, 3)  # Replace with the original image shape (height, width, channels)
combine_images(input_folder, output_file, original_shape)
