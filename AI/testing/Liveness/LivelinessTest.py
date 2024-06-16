import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import urllib
import onnxruntime
from PIL import Image
from torchvision import transforms as T
import progressbar

# YOLO Model for Face Detection
yolo_model = YOLO('detectionModel3.pt')

# Progress bar initialization
pbar = None

# Liveness Detection Class
class LivenessDetection:
    def __init__(self, checkpoint_path: str):
        if not Path(checkpoint_path).is_file():
            print("Downloading the DeepPixBiS onnx checkpoint:")
            urllib.request.urlretrieve(
                "https://github.com/ffletcherr/face-recognition-liveness/releases/download/v0.1/OULU_Protocol_2_model_0_0.onnx",
                Path(checkpoint_path).absolute().as_posix(), show_progress
            )
        self.deepPix = onnxruntime.InferenceSession(
            checkpoint_path, providers=["CPUExecutionProvider"]
        )
        self.trans = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, face_arr: np.ndarray) -> float:
        face_rgb = cv2.cvtColor(face_arr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.trans(face_pil).unsqueeze(0).detach().cpu().numpy()
        output_pixel, output_binary = self.deepPix.run(
            ["output_pixel", "output_binary"], {"input": face_tensor.astype(np.float32)}
        )
        liveness_score = (
            np.mean(output_pixel.flatten()) + np.mean(output_binary.flatten())
        ) / 2.0
        return liveness_score

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

# Initialize Liveness Detection
liveness_detector = LivenessDetection("checkpoint.onnx")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Make predictions with YOLO model for face detection
    results = yolo_model(frame)

    # Process each detected face
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract the face from the frame
            face = frame[y1:y2, x1:x2]

            # Perform liveness detection
            liveness_treshold = 0.03
            liveness_score = liveness_detector(face)
            liveness_label = "Real" if liveness_score > liveness_treshold else "Fake"
            color = (0, 255, 0) if liveness_score > liveness_treshold else (0, 0, 255)

            # Draw bounding box and liveness score on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{liveness_label} ({liveness_score:.2f})'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Webcam Liveness Detection', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
