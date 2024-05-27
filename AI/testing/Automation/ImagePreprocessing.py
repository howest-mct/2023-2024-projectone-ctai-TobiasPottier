#region Preprocessing
import os
import torch
from PIL import Image
import cv2
import os
from ultralytics import YOLO
import numpy as np

model = YOLO('./detectionModel2.pt')
def detect_and_crop_faces(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Load the image
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # Perform face detection
            results = model(image)

            # Load the image using OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Process the results and show only the detected faces
            index = 1
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Cut out the face from the frame
                    face = cv_image[y1:y2, x1:x2]

                    # Save the cropped face image
                    face_filename = f"{os.path.splitext(filename)[0]}_face_{index}.jpg"
                    face_path = os.path.join(output_folder, face_filename)
                    cv2.imwrite(face_path, face)

                    print(f"Saved cropped face to {face_path}")
                    index+=1

#endregion 

#region FindFolder
def get_highest_index_folder_path(parent_directory):
    """Get the folder path with the highest index inside the parent directory."""
    folders = [f for f in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, f))]

    if not folders:
        raise FileNotFoundError("No folders found in the specified directory.")

    # Convert folder names to integers and find the highest index
    folders.sort(key=int)
    highest_index_folder = folders[-1]

    return os.path.join(parent_directory, highest_index_folder)

def check_and_create_ready_file(folder_path):
    """Check for READY.txt file in the specified folder and create it if not present."""
    ready_file_path = os.path.join(folder_path, 'READY.txt')
    ImagesPath = os.path.join(folder_path, 'Unfiltered')

    if not os.path.exists(ready_file_path):
        print("READY.txt does not exist in the folder.")
        # Perform any other actions needed if READY.txt does not exist
        if os.path.exists(ImagesPath):
            detect_and_crop_faces(input_folder=ImagesPath, output_folder=folder_path)
        else:
            print('ERROR: READY.txt is not there, but Unfiltered folder is also not present')
        # Create the READY.txt file
        with open(ready_file_path, 'w') as f:
            f.write("This is the READY.txt file.")
        print("READY.txt file has been created.")
    else:
        print("READY.txt already exists in the folder.")
        return



def main():
    parent_directory = "C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubDataset"  # Replace with the path to your parent directory

    try:
        highest_index_folder_path = get_highest_index_folder_path(parent_directory)
        print(f"Highest index folder: {highest_index_folder_path}")
        check_and_create_ready_file(highest_index_folder_path)
    except Exception as e:
        print(f"An error occurred: {e}")

#endregion

if __name__ == "__main__":
    main()
