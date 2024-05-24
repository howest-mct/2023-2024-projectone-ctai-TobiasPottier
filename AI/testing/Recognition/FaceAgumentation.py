import os
import cv2
import imgaug.augmenters as iaa
from glob import glob

# Function to augment images
def augment_images(image_paths, augmenters, output_dir):
    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Apply augmentations
        for augmenter in augmenters:
            augmented_image = augmenter(image=image)
            # Create a new file name
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            new_image_path = os.path.join(output_dir, f"{name}_aug_{augmenter.name}{ext}")
            
            # Save augmented image
            cv2.imwrite(new_image_path, augmented_image)
            print(f"Saved augmented image: {new_image_path}")

# Define the augmenters
brightness_augmenter = iaa.Multiply((0.5, 1.5)).to_deterministic()  # Multiply the pixel values by a random value between 0.5 and 1.5
rotation_augmenter = iaa.Affine(rotate=(-25, 25)).to_deterministic()  # Rotate images by a random degree between -25 and 25

augmenters = [brightness_augmenter, rotation_augmenter]

# Directory containing the images
image_dir = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubDataset/Yahya'  # Change to your folder path

# Get all image paths
image_paths = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpeg"))

# Augment images and save them
augment_images(image_paths, augmenters, image_dir)

