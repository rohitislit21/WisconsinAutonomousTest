import cv2
import numpy as np

# Load the image from the given path
image_path = "originalImage.png"  # Ensure this path is correct
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()  # Exit the script if the image cannot be loaded
else:
    print("Image loaded successfully!")

# Convert the image to grayscale for easier edge detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Image converted to grayscale.")

# Apply GaussianBlur to smooth the image and reduce noise
smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
print("Gaussian blur applied.")

# Detect edges using the Canny edge detector
edges = cv2.Canny(smoothed_image, 100, 200)
print("Edge detection completed.")

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours found: {len(contours)}")

for i, contour in enumerate(contours):
    accuracy = 0.01 * cv2.arcLength(contour, True)
    approx_curve = cv2.approxPolyDP(contour, accuracy, True)

    if len(approx_curve) == 4:
        print(f"Drawing contour {i+1} with 4 points.")
        cv2.drawContours(image, [approx_curve], 0, (0, 0, 255), 2)

output_path = "answer.png"
saved = cv2.imwrite(output_path, image)

if saved:
    print(f"Image saved successfully at {output_path}")
else:
    print("Error: Failed to save image")
