import cv2
import numpy as np

# Load the image of the empty space
empty_space_path = 'empty_space_screenshot.png'  # Update this path
empty_space_img = cv2.imread(empty_space_path, cv2.IMREAD_GRAYSCALE)

# Calculate the standard deviation of the grayscale image
std_dev = np.std(empty_space_img)

# Set the threshold slightly above the standard deviation
# You may need to adjust the multiplier based on the actual variation in your empty tiles
threshold = std_dev * 1.1  # for example, 10% above the standard deviation

# Print the standard deviation and the chosen threshold
print(f"Standard Deviation of Empty Space: {std_dev}")
print(f"Chosen Threshold: {threshold}")
