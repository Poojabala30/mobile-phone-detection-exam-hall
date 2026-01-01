from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Read image
image = cv2.imread("exam_hall.jpg")

if image is None:
    print("ERROR: Image not found")
    exit()

# Run detection
results = model(image)

# Loop through detections
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    # COCO class ID for cell phone = 67
    if class_id == 67 and confidence > 0.3:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"Mobile Phone {confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

# Show output
cv2.imshow("Mobile Phone Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
