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
from LivenessDetection import LivenessDetection
#endregion


#region BLE connection
print('Starting BLE Client...')
tx_q = queue.Queue()
rx_q = queue.Queue()
BLE_DEVICE_MAC = "D8:3A:DD:D9:73:57"
connection_event = Event()

def init_ble_thread():
    global ble_client_thread
    try:
        # Creating a new thread for running a function 'run' with specified arguments.
        ble_client_thread = Thread(target=BLE_client.run, args=(
            rx_q, tx_q, None, BLE_DEVICE_MAC, connection_event), daemon=True)
        # Starting the thread execution.
        ble_client_thread.start()
    except Exception as e:
        print(f"Error starting BLE client thread: {e}")

# Initialize the thread variable
ble_client_thread = None

# # Repeat the thread initialization until the connection_event is set
# while not connection_event.is_set():
#     if ble_client_thread is None or not ble_client_thread.is_alive():
#         init_ble_thread()
#     connection_event.wait(timeout=5)  # Optional timeout to avoid tight looping

print('BLE CONNECTION ESTABLISHED')
#endregion

#region flagging and data retrieval

FLAG_FILE_PATH = './flag/reload_flag.txt'
CURRENT_CLASSIFIER_PATH = 'SVM_classifier.pkl'
def reload_resources():
    global classifier, label_mapping, auth_labels, known_embeddings, sequential_list_labels, reloading_stop
    reloading_stop = True
    time.sleep(.2) # wait for main loop to stop
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
        time.sleep(1)  # Check every second

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
# load the liveness_detector
liveness_detector = LivenessDetection("checkpoint.onnx")
print('Models Succesfully Loaded!')


def main(take_picture_event, show_face_event, face_det_event, stop_event, camera_open_event):
    accept_frame_event = Event()
    rx_frame = queue.Queue(maxsize=1)  # Limit the queue size to 1 to ensure only the latest frame is processed
    tx_frame = queue.Queue(maxsize=1)

    #region Calculate Frame
    def CalculateRecogntion():
        accept_frame_event.set()
        while not stop_event.is_set():
            if not reloading_stop:
                frame = rx_frame.get()
                if frame is not None:
                    send_dict = {}
                    results = yolo_model(frame)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            user_auth = False
                            is_person = False
                            info_list = []
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # Cut out the face from the frame
                            face = frame[y1:y2, x1:x2]

                            liveness_treshold = 0.05
                            liveness_score = liveness_detector(face)
                            if liveness_score > liveness_treshold:
                                is_person = True
                                
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
                                confidence_threshold = 0.60
                                euclidean_distance_treshold = 0.70

                                if confidence > confidence_threshold and min_distance < euclidean_distance_treshold and predicted_label in auth_labels:
                                    user_auth = True
                                info_list = [is_person, user_auth, predicted_label, label_mapping[predicted_label], confidence, min_distance, liveness_score]
                            else:
                                info_list = [is_person, liveness_score]
                                    
                            send_dict[box] = info_list
                    if tx_frame.full():
                        tx_frame.get()  # Remove the old frame if the queue is full
                    tx_frame.put(send_dict)
                    if rx_frame.empty():
                        accept_frame_event.set()



    calc_thread = Thread(target=CalculateRecogntion, daemon=True)
    calc_thread.start()

    #region CameraDisplay
    print('Opening Camera...')
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        raise Exception("Could not open camera")

    # clear the queue manually
    def clear_queue(q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    #last user detected state
    last_state = None
    captures_dir = './captures'
    print('Program Ready!')
    camera_open_event.set()
    face_count = 0
    face_window_open = False
    latest_calc_dict = None
    while not stop_event.is_set():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            raise Exception("Can't receive camera frame, exiting...")
        if not reloading_stop:   # make sure to only make predictions when models and SQL data are not being reloaded
            if show_face_event.is_set():
                results = yolo_model(frame)
                face_detected = False
                face_count = 0
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Cut out the face from the frame
                        face = frame[y1:y2, x1:x2]
                        face_detected = True

                        face_count += 1
                        if face_count == 1:
                            cv2.imshow('Detected Face', face)
                            face_window_open = True
                            if take_picture_event.is_set():
                                # Save the current frame to /captures directory
                                os.makedirs(captures_dir, exist_ok=True)
                                existing_pictures = [f for f in os.listdir('./captures') if f.startswith("Fpic")]
                                if existing_pictures:
                                    indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_pictures]
                                    next_index = max(indices) + 1
                                else:
                                    next_index = 1
                                capture_path = os.path.join(captures_dir, f'Fpic_{next_index}.jpg')
                                cv2.imwrite(capture_path, face)
                                print(f"FacePicture saved to {capture_path}")

                                # Clear the take_picture_event
                                take_picture_event.clear()
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            cv2.putText(frame, f'Face Picture', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                if face_detected:
                    face_det_event.set()
                else:
                    face_det_event.clear()
            elif face_window_open:
                cv2.destroyWindow('Detected Face')
                face_window_open = False
            else:
                if accept_frame_event.is_set() and not rx_frame.full():
                    rx_frame.put(frame)
                    accept_frame_event.clear()
                if not tx_frame.empty():
                    latest_calc_dict: dict = tx_frame.get()
                box_states = []
                if latest_calc_dict is not None:
                    for key, value in latest_calc_dict.items():
                        x1, y1, x2, y2 = key.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if value[0] == True:
                            predicted_label_str = f"{value[2]} (={value[3]}) | ({value[4]:.2f})  DIST {value[5]:.2f} | LIV {value[6]:.2f}"

                            if value[1] == True: # is the person authorized
                                cv2.putText(frame, f'{predicted_label_str}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                current_state = 'UD'
                            else:
                                output_str = "Non-User"
                                cv2.putText(frame, f'{output_str}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                current_state = 'NUD'
                        else:
                            output_str = f"Not A Person! | LIV {value[1]:.2f}"
                            cv2.putText(frame, f'{output_str}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
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
    clear_queue(tx_q)
    tx_q.put('exit')
    stop_event.clear()
    camera_open_event.clear()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('Script called from __main__, no action taken')

#endregion