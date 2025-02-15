from tifffile import imread
import matplotlib.pyplot as plt


# Read multi-channel TIFF
image = imread(r"C:\Vicky\Tu Braunschweig\Semester 5\FPDS\DLR_Project\Train image 1\train\part_7.tif")

# # Display a specific channel (e.g., channel 0)
# plt.imshow(image[0], cmap='viridis')
# plt.title('Channel 0 - Semantic Segmentation')
# plt.colorbar()
# plt.show()


from tifffile import imread, imwrite
import numpy as np


# Enhance the visibility of labels
# Normalize or threshold the image if it contains valid data
normalized = (image / np.max(image) * 255).astype(np.uint8)

# Save the enhanced TIFF
imwrite(r"C:\Vicky\Tu Braunschweig\Semester 5\FPDS\DLR_Project\Train image 1\train\part_7_enhanced.tif", normalized)




####################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Correct file path for your TIFF file
file_path = r"C:\Vicky\Tu Braunschweig\Semester 5\FPDS\DLR_Project\Train image 1\train\part_7_enhanced.tif"  # Ensure the file path is correct and accessible
output_file = r"C:\Vicky\Tu Braunschweig\Semester 5\FPDS\DLR_Project\Train image 1\train\part_7_label.tif"   # Desired output file name

# Load the TIFF file
segmentation = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

# Check if the file is loaded correctly
if segmentation is None:
    raise FileNotFoundError(f"Error: Unable to read file '{file_path}'. Please check the file path or integrity.")

# Define the new color mapping
# Mapping: 0 -> White, 255 -> Yellow, 85 -> Blue, 170 -> Red
color_mapping = {
    0: [255, 255, 255],   # White for background
    255: [0, 255, 255],   # Yellow for "others"
    85: [0, 0, 255],      # Blue for "90 degrees"
    170: [255, 0, 0]      # Red for "0 degrees"
}

# Create a blank RGB image for the colorized segmentation
height, width = segmentation.shape
colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)

# Apply the color mapping
for grayscale_value, rgb_color in color_mapping.items():
    colored_segmentation[segmentation == grayscale_value] = rgb_color

# Save the resulting colorized image
cv2.imwrite(output_file, colored_segmentation)

# Optional: Display the image using matplotlib
plt.imshow(cv2.cvtColor(colored_segmentation, cv2.COLOR_BGR2RGB))
plt.title("Colorized Segmentation")
plt.axis("off")
plt.show()

print(f"Colorized segmentation saved as: {output_file}")








