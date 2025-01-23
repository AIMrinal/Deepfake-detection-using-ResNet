import cv2
import numpy as np
import os
import random
import exifread
from tqdm import tqdm

def auto_orient_image(image_path):
    # Get orientation information from EXIF data
    orientation = None
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
        if 'Image Orientation' in tags:
            orientation = tags['Image Orientation'].values[0]

    # Load the image
    image = cv2.imread(image_path)

    # Rotate the image based on orientation
    if orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image

def resize_image(image, size=(640, 640)):
    # Resize image to specified size
    resized_image = cv2.resize(image, size)
    return resized_image

def random_rotation(image, max_angle=15):
    # Randomly rotate the image between -max_angle and +max_angle degrees
    angle = random.uniform(-max_angle, max_angle)
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def random_brightness(image, max_delta=30):
    # Randomly adjust brightness of the image
    delta = random.uniform(-max_delta, max_delta)
    adjusted_image = np.clip(image + delta, 0, 255).astype(np.uint8)
    return adjusted_image

def salt_and_pepper_noise(image, amount=0.01):
    # Add salt and pepper noise to the image
    noisy_image = np.copy(image)
    num_pixels = int(amount * image.size)
    salt_coords = [np.random.randint(0, i - 1, num_pixels // 2) for i in image.shape]
    pepper_coords = [np.random.randint(0, i - 1, num_pixels // 2) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    return noisy_image

def preprocess(input_image_path, output_path):
    original_image = cv2.imread(input_image_path)

    # Preprocess and augment the image
    original_image = auto_orient_image(input_image_path)
    original_image = resize_image(original_image)

    for i in range(2):
        # Apply augmentations
        augmented_image = random_rotation(original_image)
        augmented_image = random_brightness(augmented_image)
        augmented_image = salt_and_pepper_noise(augmented_image)
    
        # Save augmented image with a unique name
        output_image_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(input_image_path))[0]}_{i+1}.jpg")
        cv2.imwrite(output_image_path, augmented_image)

# Input directory containing images to preprocess
input_dir = "/Users/mrinal/Desktop/Datasets/real_vs_fake/train/real"

# Output directory to save preprocessed images
output_dir = "/Users/mrinal/Desktop/Datasets/Preprocessed/Ds-1/Real"
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Process each image in the input directory with progress bar
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_image_path = os.path.join(input_dir, filename)
        preprocess(input_image_path, output_dir)

print("Preprocessing and augmentation completed.")
