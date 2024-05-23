import os
import shutil
from random import sample

def create_subset(input_image_dir, input_label_dir, output_image_dir, output_label_dir, subset_size):
    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Get a list of all image files
    all_images = os.listdir(input_image_dir)
    
    # Randomly select a subset of image files
    subset_images = sample(all_images, subset_size)
    
    for image_file in subset_images:
        # Copy image file
        shutil.copy(os.path.join(input_image_dir, image_file), os.path.join(output_image_dir, image_file))
        
        # Copy corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        shutil.copy(os.path.join(input_label_dir, label_file), os.path.join(output_label_dir, label_file))

# Define paths
input_image_dir = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetDetection/Dataset/images/train'
input_label_dir = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetDetection/Dataset/labels/train'
output_image_dir = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetDetection/SubSetDataset/images/train'
output_label_dir = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetDetection/SubSetDataset/labels/train'
subset_size = 1000  # desired number of images

# Create subset
create_subset(input_image_dir, input_label_dir, output_image_dir, output_label_dir, subset_size)

# Repeat for validation set
input_image_dir = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetDetection/Dataset/images/val'
input_label_dir = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetDetection/Dataset/labels/val'
output_image_dir = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetDetection/SubSetDataset/images/val'
output_label_dir = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetDetection/SubSetDataset/labels/val'
subset_size = 500  # desired number of validation images

create_subset(input_image_dir, input_label_dir, output_image_dir, output_label_dir, subset_size)