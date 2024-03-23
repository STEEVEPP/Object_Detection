import cv2
from ultralytics import YOLO
import numpy
#from ultralytics.solutions import distance_calculation

model = YOLO('yolov9e.pt')

org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
 
# Using cv2.putText() method 
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
        
        print(class_name) 

        conf = box.conf[0].item()  
        
        # Print the detection result
        print(f"{class_name} detected at ({x1}, {y1}), ({x2}, {y2}) with confidence {conf}")

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        org = (int(x2), int(y1))

        font = cv2.FONT_HERSHEY_SIMPLEX 

        l1 = str(label)
        print(l1)

        image = cv2.putText(frame, class_name, org, font,  fontScale, color, thickness, cv2.LINE_AA) 


    # Display the frame with the detected objects
    cv2.imshow('Object Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1)==27:
        break

# Release the camera and destroy all windows
#cap.release()
#cv2.destroyAllWindows()