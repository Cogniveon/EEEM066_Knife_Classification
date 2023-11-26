import os
import random
import cv2
import numpy as np
import argparse

def rotate_image(image, angle):
    """Rotate the input image by a given angle."""
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_NEAREST)
    return rotated_image

def adjust_brightness(image, factor):
    """Adjust the brightness of the input image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * factor, 0, 255).astype(np.uint8)
    brightened_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return brightened_image

def augment_image(image_path):
  filedir = os.path.dirname(image_path)
  filename = os.path.basename(image_path).split('.')[0]
  ext = os.path.basename(image_path).split('.')[1]

  original_image = cv2.imread(image_path)

  if original_image is None:
    print(f"Error: Could not read the image from {image_path}")
    return

  rotation_angles = [-90, -30, 30, 0, 90, 180]
  brightness_values = [0.5, 0.7, 1.5]

  for angle in rotation_angles:
     for brightness in brightness_values:
        rotated_image = rotate_image(original_image, angle)

        augmented_image = adjust_brightness(rotated_image, brightness)

        output_path = os.path.join(filedir, f"{filename}_r{angle}_b{brightness}.{ext}")

        print(f"{output_path},192")
        cv2.imwrite(output_path, augmented_image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Creates augmented version of the input image.")
  parser.add_argument("image_path", help="Path to the image file.")
  args = parser.parse_args()

  augment_image(args.image_path)