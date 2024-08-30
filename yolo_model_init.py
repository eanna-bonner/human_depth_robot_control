#  Global imports, functions and variables
import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.4
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def process_depth(frame, depth_image, xmin, ymin, xmax, ymax):       

    box_center_depth = (int((((ymin + ymax) // 2)-(ymax-ymin)//4)//(8/3)), int(((xmin + xmax) // 2)//3.2)) 
    # Get depth data for the bounding box
    depth_data = depth_image[box_center_depth[0]-3:box_center_depth[0]+3, box_center_depth[1]-3:box_center_depth[1]+3]
    # remove null depth data
    non_zero = depth_data[depth_data != 0]
    # calculate average depth
    avg_depth = int(np.mean(non_zero)) if len(non_zero) > 0 else 0
    # print("avg depth",avg_depth)

    cv2.putText(frame, f"Depth: {avg_depth}mm", (xmin, ymin-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    return frame, avg_depth

# YOLO Model
from ultralytics import YOLO
model = YOLO("./models/yolov8n.pt")



def visualize_YOLO(image, detections, depth_image):
    depths = []
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), GREEN, 2)

        box_center = (int((xmin + xmax) // 2), int((ymin + ymax) // 2)-(ymax-ymin)//4)
        cv2.circle(image, box_center, 5, RED, -1)

        image, avg_depth = process_depth(image, depth_image, xmin, ymin, xmax, ymax)
        depths.append(avg_depth)
    
    closest_det = min(depths) if len(depths) > 0 else 3000
    
    return image, closest_det

def get_yolo_detections(frame):
    return model(frame, classes=0, verbose=False)[0] # run the model

def run_yolo(frame, depth_frame):
    detections = get_yolo_detections(frame)
    frame, closest_det = visualize_YOLO(frame, detections, depth_frame)
    return frame, closest_det

