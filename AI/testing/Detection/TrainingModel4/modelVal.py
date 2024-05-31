from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("./runs/detect/train/weights/best.pt")

# Evaluate the model on the validation dataset
results = model.val(data="data.yaml", imgsz=640)

# Print the evaluation results
print("Validation Results:")
print(results)