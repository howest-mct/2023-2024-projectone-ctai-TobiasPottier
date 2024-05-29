#region imports
print('Importing Libraries...')
import queue
import BLE_client
from threading import Event, Thread
import mysql.connector
import time
import pickle
import os
import cv2
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
from scipy.spatial.distance import euclidean
#endregion


#region BLE connection
print('Starting BLE Client...')
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
print('BLE CONNECTION ESTABLISHED')
#endregion

#region flagging and data retrieval

FLAG_FILE_PATH = './flag/reload_flag.txt'
CURRENT_CLASSIFIER_PATH = 'SVM_classifier.pkl'
def reload_resources():
    global classifier, label_mapping, auth_labels, known_embeddings, sequential_list_labels, reloading_stop
    reloading_stop = True
    time.sleep(.1) # wait for main loop to stop
    # Load the trained classifier
    with open(CURRENT_CLASSIFIER_PATH, 'rb') as f:
        classifier = pickle.load(f)

    # Reload SQL data
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

    def retrieve_auth_labels():
        cursor.execute("SELECT LabelID FROM auth")
        results = cursor.fetchall()
        return [label[0] for label in results]
    
    def retrieve_known_embeddings():
        cursor.execute("SELECT embedding, label FROM face_data")
        results = cursor.fetchall()
        embeddings_dict = {}
        for (binary_embedding, label) in results:
            embedding = pickle.loads(binary_embedding)
            if label not in embeddings_dict:
                embeddings_dict[label] = []
            embeddings_dict[label].append(embedding)
        return embeddings_dict

    label_mapping = retrieve_label_mapping()
    auth_labels = retrieve_auth_labels()
    known_embeddings = retrieve_known_embeddings()
    
    # Create the sequential list based on the number of keys in label_mapping
    sequential_list_labels = list(label_mapping.keys())
    reloading_stop = False

    print(f'Labels: {label_mapping}')
    print(f'Auth: {auth_labels}')
    print(f'Label List: {sequential_list_labels}')
    cursor.close()
    conn.close()
    print("Resources reloaded.")

# Function to monitor the flag file
def monitor_flag_file():
    while True:
        if os.path.exists(FLAG_FILE_PATH):
            reload_resources()
            os.remove(FLAG_FILE_PATH)
        time.sleep(1)  # Check every 5 seconds

# Start monitoring the flag file in a separate thread
flag_monitor_thread = Thread(target=monitor_flag_file, daemon=True)
flag_monitor_thread.start()

print('Getting Data from SQL Database...')
reload_resources()
#endregion

#region Face Recognition

print('Loading Models...')
# Load the YOLO model for face detection
yolo_model = YOLO('./detectionModel3.pt')

# Load the FaceNet model for face embedding
embedder = FaceNet()

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
    if not reloading_stop:   # make sure to only make predictions when models and SQL data are not being reloaded
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
                predicted_label = sequential_list_labels[max_index]

                # Compute the Euclidean distance to known faces
                known_embeddings_for_label = known_embeddings[predicted_label]
                distances = [euclidean(embeddings[0], emb) for emb in known_embeddings_for_label]
                min_distance = min(distances)

                # Only display the label if confidence is above a certain threshold
                confidence_threshold = 0.50
                euclidean_distance_treshold = 0.70
                if confidence > confidence_threshold and min_distance < euclidean_distance_treshold:
                    # Convert the predicted label to a string
                    predicted_label_str = f"{predicted_label} (={label_mapping[predicted_label]}) | ({confidence:.2f})  DIST {min_distance:.2f}"

                    # Draw bounding box and label on the frame
                    cv2.putText(frame, f'{predicted_label_str}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                    if predicted_label in auth_labels:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        current_state = 'UD'
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        current_state = 'NUD'
                    
                else:
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'Non-User | ({confidence:.2f}) DIST {min_distance:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
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