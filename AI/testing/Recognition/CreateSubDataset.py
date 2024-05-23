import os
import shutil
import random

def create_subset_dataset(input_dir, output_dir, num_folders):
    # Get the list of all folders in the input directory
    all_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Randomly select the specified number of folders
    selected_folders = random.sample(all_folders, num_folders)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the selected folders and their contents to the output directory
    for folder in selected_folders:
        src_folder = os.path.join(input_dir, folder)
        dest_folder = os.path.join(output_dir, folder)
        shutil.copytree(src_folder, dest_folder)
    
    print(f"Created subset dataset with {num_folders} folders in '{output_dir}'.")

# Define the paths
input_directory = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/dataset'
output_directory = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubDataset'
number_of_folders = 200  # Number of folders to be in the subset

# Create the subset dataset
create_subset_dataset(input_directory, output_directory, number_of_folders)
