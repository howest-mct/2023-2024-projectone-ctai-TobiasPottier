import cv2
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
import pickle
import os

# Load the YOLO model for face detection
yolo_model = YOLO('./best.pt')

# Load the FaceNet model for face embedding
embedder = FaceNet()

# Load the trained classifier and label encoder
with open('SVM_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)


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

    # Make predictions with YOLO model
    results = yolo_model(frame)

    # Process the results and show only the detected faces
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Cut out the face from the frame
            face = frame[y1:y2, x1:x2]

            # Resize the face to 160x160 as required by FaceNet
            face = cv2.resize(face, (160, 160))
            face = face.astype('float32')
            face = np.expand_dims(face, axis=0)

            # Generate embeddings for the face
            embeddings = embedder.embeddings(face)

            probabilities = classifier.predict_proba(embeddings)
            max_index = np.argmax(probabilities)
            confidence = probabilities[0][max_index]
            predicted_label = max_index

            # Only display the label if confidence is above a certain threshold
            confidence_threshold = 0.3
            if confidence > confidence_threshold:
                # Convert the predicted label to a string
                predicted_label_str = f"{predicted_label} ({confidence:.2f})"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{predicted_label_str}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
               
                if predicted_label == 1:
                    os.system('cls')
                    print('UNLOCK')

    # Display the resulting frame
    cv2.imshow('Webcam Face Detection and Recognition', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
