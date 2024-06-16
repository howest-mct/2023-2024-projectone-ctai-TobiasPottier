# MTCCT = Multi-Threaded-Camera-Calculation-Technique

print('Importing Libraries')
import cv2
from ultralytics import YOLO
import time
import threading
import queue

print('Importing Model')
model = YOLO('detectionModel3.pt')

print('Opening Camera')
cap = cv2.VideoCapture(0)

accept_frame_event = threading.Event()
rx = queue.Queue(maxsize=1)  # Limit the queue size to 1 to ensure only the latest frame is processed
tx = queue.Queue(maxsize=1)


def CalculateDetection(rx: queue.Queue, tx: queue.Queue, accept_frame_event: threading.Event):
    accept_frame_event.set()
    while True:
        frame = rx.get()
        if frame is not None:
            results = model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    time.sleep(.2) # simulated calc time
            if tx.full():
                tx.get()  # Remove the old frame if the queue is full
            tx.put(results)
            if rx.empty():
                accept_frame_event.set()


calc_thread = threading.Thread(target=CalculateDetection, args=(rx, tx, accept_frame_event), daemon=True)
calc_thread.start()

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

latest_calc = None
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    if accept_frame_event.is_set() and not rx.full():
        rx.put(frame)
        accept_frame_event.clear()

    if not tx.empty():
        latest_calc = tx.get()

    if latest_calc is not None:
        for result in latest_calc:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f'{box.conf[0]:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
   
    cv2.imshow('Webcam Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
