from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data = "pothole/data.yaml",
    epochs = 25,
    imgsz = 640,
    batch = 16,
    name = "pothole_yolov8m"
)
# Load best trained model
trained_model = YOLO("runs/detect/pothole_yolov8m/weights/best.pt")

# Save a copy for deployment
trained_model.save("pothole_detector.pt")