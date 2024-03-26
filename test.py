import cv2
from ultralytics import YOLO
import numpy
#from ultralytics.solutions import distance_calculation
from cvzone.FaceMeshModule import FaceMeshDetector
import pyttsx3
import threading
import datetime


detector = FaceMeshDetector(maxFaces=50)
current_time = datetime.datetime.now().time()
engine = pyttsx3.init()

model = YOLO('yolov8s.pt')

org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
distance1 = 0
# Using cv2.putText() method 
# Open the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

cell_object_width= 15
cell_d = 46
bottle_object_width = 35
bottle_d = 46
kb_object_width = 30.75
kb_d = 46
book_object_width = 14
book_d = 46


prev_frame_time = 0
new_frame_time = 0

def calculate_distance(cell_object_width, cell_d, w):
            cell_focal_length = (cell_object_width * cell_d) / w
            cell_apparent_width = w  # Apparent width of the cell phone in pixels
            cell_distance = ((cell_object_width * 3.9) / cell_apparent_width)*100
            return cell_distance

def speak(text):
    engine.setProperty('rate', 150)  # Adjust the speaking rate (words per minute)
    engine.setProperty('volume', 0.9)  # Set the volume level (0.0 to 1.0)
    engine.setProperty('pitch', 50) 
    engine.say(text)
    engine.runAndWait()

if current_time >= datetime.time(5, 0, 0) and current_time < datetime.time(12, 0, 0):
    speak("Good morning")
elif current_time >= datetime.time(12, 0, 0) and current_time < datetime.time(18, 0, 0):
    speak("Good afternoon")
else:
    speak("Good evening")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame, show=False, save=False)

    # Process the results
    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0]
        w, h = x2 - x1, y2 - y1
        label = box.cls[0].item()  # Get the label of the first box

        class_name = model.names[int(label)]
        
        print(class_name) 

        conf = box.conf[0].item()  
        # Hypothetical function to calculate distance

        if(class_name == "cell phone"):
            distance1 = calculate_distance(cell_object_width,cell_d,w)
            class_name = f"{class_name} " + str(int(distance1)) + " cm"
        elif(class_name == "bottle"):
            distance1 = calculate_distance(bottle_object_width,bottle_d,w)
            class_name = f"{class_name} " + str(int(distance1)) + " cm"
        elif(class_name == "keyboard"):
            distance1 == calculate_distance(kb_object_width,kb_d,w)
            class_name = f"{class_name} " + str(int(distance1)) + " cm"
        elif(class_name == "book"):
            distance1 == calculate_distance(book_object_width,book_d,w)
            class_name = f"{class_name} " + str(int(distance1)) + " cm"
        elif(class_name == "person"):
            frame,faces = detector.findFaceMesh(frame,draw=False)
            if faces:
                face=faces[0]
                pointLeft = face[145]
                pointRight = face[374]
                w,_ = detector.findDistance(pointLeft,pointRight)
                W = 6.3
                f = 668
                distance1 = (W*f)/w
                class_name = f"{class_name} " + str(int(distance1)) + " cm"
        else:
            class_name = class_name
        
        if(distance1 <= 50):
            
            text ="Distance is too Low! maintain the distance"
            threading.Thread(target=speak, args=(text,)).start()
            frame = cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)

        # Print the detection result
        print(f"{class_name} detected at ({x1}, {y1}), ({x2}, {y2}) with confidence {conf}")

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        org = (int(x1), int(y1))
        
        font = cv2.FONT_HERSHEY_SIMPLEX 

        l1 = str(label)
        print(l1)

        image = cv2.putText(frame, class_name, org, font,  fontScale, color, thickness, cv2.LINE_AA) 


    # Display the frame with the detected objects
    cv2.imshow('Object Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1)==27:
        speak("Thank You")
        break