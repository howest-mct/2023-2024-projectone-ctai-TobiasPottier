import os
import pandas as pd

def create_labels_csv(input_dir, output_csv):
    # List to hold the image paths and labels
    data = []

    # Traverse through each folder
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, image_name)
                    data.append([image_path, folder])
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    df.to_csv(output_csv, index=False)
    print(f"Labels CSV created at: {output_csv}")

# Define the paths
input_directory = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubDataset'
output_csv_file = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubLabels/labels.csv'

# Create the labels CSV
create_labels_csv(input_directory, output_csv_file)
