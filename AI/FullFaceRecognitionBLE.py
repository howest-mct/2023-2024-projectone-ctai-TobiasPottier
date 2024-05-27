
#region BLE connection
print('Starting BLE Client...')
import queue
import BLE_client
from threading import Event, Thread
tx_q = queue.Queue()
rx_q = queue.Queue()
BLE_DEVICE_MAC = "D8:3A:DD:D9:73:57"
connection_event = Event()

# def init_ble_thread():
#     # Creating a new thread for running a function 'run' with specified arguments.
#     ble_client_thread = Thread(target=BLE_client.run, args=(
#         rx_q, tx_q, None, BLE_DEVICE_MAC, connection_event), daemon=True)
#     # Starting the thread execution.
#     ble_client_thread.start()
# init_ble_thread()
# connection_event.wait()

#endregion
print('BLE CONNECTION ESTABLISHED')


#region Retrieve Labels from MySQL
import mysql.connector
print('Connecting to MySQL...')
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='face_recognition'
)
cursor = conn.cursor()

def retrieve_label_mapping():
    cursor.execute("SELECT LabelID, FaceName FROM face_name")
    results = cursor.fetchall()
    return {label: name for label, name in results}

label_mapping = retrieve_label_mapping()
print('Label mapping retrieved from MySQL:', label_mapping)

cursor.close()
conn.close()
#endregion


#region Face Recognition
import cv2
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
import pickle
import os


print('Loading Models...')
# Load the YOLO model for face detection
yolo_model = YOLO('./detectionModel2.pt')

# Load the FaceNet model for face embedding
embedder = FaceNet()

# Load the trained classifier and label encoder
with open('SVM_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

print('Models Succesfully Loaded!')
print('Opening Camera...')
# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# clear the queue manually
def clear_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

#last user detected state
last_state = None
print('Program Ready!')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Make predictions with YOLO model
    results = yolo_model(frame)

    current_state = 'NUD'  # Default state is No User Detected

    # Process the results and show only the detected faces
    for result in results:
        boxes = result.boxes
        box_states = []
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

            probabilities = classifier.predict_proba(embeddings)  # this gives you each probability per class
            max_index = np.argmax(probabilities)  # max index = label of the person who has the highest probability
            confidence = probabilities[0][max_index]
            predicted_label = max_index

            # Only display the label if confidence is above a certain threshold
            confidence_threshold = 0.80
            if confidence > confidence_threshold:
                # Convert the predicted label to a string
                predicted_label_str = f"{predicted_label} (={label_mapping[predicted_label]}) | ({confidence:.2f})"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{predicted_label_str}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
               
                if predicted_label == 1:
                    current_state = 'UD'
                else:
                    current_state = 'NUD'
                
            else:
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Non-User', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                current_state = 'NUD'

            box_states.append(current_state)
        current_state = 'UD' if 'UD' in box_states else 'NUD'

    # Send the current state if it has changed from the last state
    if current_state != last_state:
        clear_queue(tx_q)
        tx_q.put(current_state)
        last_state = current_state


    # Display the resulting frame
    cv2.imshow('Webcam Face Detection and Recognition', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
#endregion