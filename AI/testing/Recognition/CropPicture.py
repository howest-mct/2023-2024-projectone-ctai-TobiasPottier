import torch
from PIL import Image
import cv2
import os
from ultralytics import YOLO
import numpy as np

model = YOLO('./best.pt')

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

# Define input and output folders
input_folder = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubDataset/Alessia/Unfiltered'
output_folder = 'C:/1-PC_M/1AI/ProjectOne/2ProjectOneGithub/DatasetRecognition/SubDataset/Alessia'

# Run the detection and cropping
detect_and_crop_faces(input_folder, output_folder)
