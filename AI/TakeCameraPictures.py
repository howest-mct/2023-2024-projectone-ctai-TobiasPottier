import cv2
from ultralytics import YOLO
import os

# Load the trained model
model = YOLO('detectionModel3.pt')

def main(camera_opened_event, take_picture_event, stop_event, face_det_event):

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        raise Exception("Could not open webcam")

    # Set the event to signal that the camera is opened
    camera_opened_event.set()
    captures_dir = './captures'
    face = None
    while not stop_event.is_set():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Make predictions with YOLO model
        results = model(frame)

        face_detected = False

        # Process the results and show only the detected faces
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Cut out the face from the frame
                face = frame[y1:y2, x1:x2]
                face_detected = True

                # Show the cutout face
                cv2.imshow('Detected Face', face)

        if face_detected:
            face_det_event.set()
        else:
            face_det_event.clear()

        # Check if the take_picture_event is set
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

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    stop_event.clear()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print('Script called from __main__, no action taken')
