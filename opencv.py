import cv2
from ultralytics import YOLO
import numpy as np
from ultralytics.solutions import distance_calculation

model = YOLO('yolov9e.pt')

# Camera parameters
focal_length = 1000  # Assuming a focal length of 1000 mm
sensor_width = 6.2  # Sensor width in mm (e.g., for a 12MP camera)

# Object sizes in the image (in pixels)
object_sizes = {
    'person': (170, 200),
    'bicycle': (100, 150),
    'car': (200, 300),
    'motorcycle': (150, 200),
    'airplane': (300, 400),
    'bus': (250, 350),
    'train': (300, 400),
    'truck': (350, 450),
    'boat': (200, 250),
    'traffic light': (50, 100),
    'fire hydrant': (50, 100),
    'stop sign': (50, 100),
    'parking meter': (50, 100),
    'bench': (100, 200),
    'bird': (20, 50),
    'cat': (100, 150),
    'dog': (150, 200),
    'horse': (250, 300),
    'sheep': (150, 200),
    'cow': (250, 300),
    'elephant': (400, 500),
    'bear': (350, 400),
    'zebra': (300, 400),
    'giraffe': (500, 600),
    'backpack': (50, 100),
    'umbrella': (100, 150),
    'handbag': (100, 150),
    'tie': (50, 100),
    'suitcase': (250, 350),
    'frisbee': (150, 200),
    'skis': (300, 400),
    'snowboard': (300, 400),
    'sports ball': (100, 150),
    'kite': (250, 350),
    'baseball bat': (250, 350),
    'baseball glove': (150, 200),
    'skateboard': (350, 450),
    'surfboard': (350, 450),
    'tennis racket': (250, 350),
    'bottle': (100, 150),
    'wine glass': (100, 150),
    'cup': (100, 150),
    'fork': (50, 100),
    'knife': (100, 150),
    'spoon': (50, 100),
    'bowl': (150, 200),
    'banana': (100, 150),
    'apple': (100, 150),
    'sandwich': (250, 350),
    'orange': (100, 150),
    'broccoli': (100, 150),
    'carrot': (100, 150),
    'hot dog': (150, 250),
    'pizza': (250, 350),
    'donut': (150, 200),
    'cake': (250, 350),
    'chair': (250, 350),
    'couch': (450, 550),
    'potted plant': (200, 300),
    'bed': (600, 800),
    'dining table': (800, 1200),
    'toilet': (450, 550),
    'tv': (1000, 1500),
    'laptop': (500, 700),
    'mouse': (50, 100),
    'remote': (100, 150),
    'keyboard': (500, 700),
    'cell phone': (100, 150),
    'microwave': (350, 450),
    'oven': (450, 550),
    'toaster': (250, 350),
    'sink': (450, 550),
    'refrigerator': (700, 900),
    'book': (150, 200),
    'clock': (150, 200),
    'vase': (200, 300),
    'scissors': (100, 150),
    'teddy bear': (150, 200),
    'hair drier': (300, 400),
    'toothbrush': (100, 150),
}

# Function to calculate distance
def calculate_distance(height, width, focal_length, sensor_width, pixel_size):
    height = height / 1000
    width = width / 1000
    distance = (height * focal_length * sensor_width) / (width * pixel_size)
    return distance

# Initialize distance calculator
dist_obj = distance_calculation.DistanceCalculation()
dist_obj.set_args(names=model.names, view_img=True, focal_length=focal_length, sensor_width=sensor_width)

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame, show=False, save=False)

    # Process the results
    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        label = box.cls[0].item()  # Get the label of the first box

        class_name = model.names[int(label)]
        
        conf = box.conf[0].item()  

        # Print the detection result
        print(f"{class_name} detected at ({x1}, {y1}), ({x2}, {y2}) with confidence {conf}")

        # Calculate the distance
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        pixel_size = np.sqrt(width**2 + height**2)
        distance = calculate_distance(height, width, focal_length, sensor_width, pixel_size)

        # Print the distance
        print(f"Distance: {distance} m")

        # Draw the bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2),int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add distance to the list
        dist_obj.add_distance(distance, class_name)

    # Display the distance and class name of each object on the video frame
        im0 = dist_obj.start_process(frame)

    # Display the frame with the detected objects
    cv2.imshow('Object Detection', im0)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1)==27:
        break