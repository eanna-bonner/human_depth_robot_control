from rtde_control import RTDEControlInterface as RTDEControl
import keyboard
import threading
import time
import queue
import datetime
import cv2
import pykinect_azure as pykinect
from skimage.color import gray2rgb

# YOLO Model
from ultralytics import YOLO
model = YOLO("./models/yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.4
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED

# Start device
device = pykinect.start_device(config=device_config)

# cv2 Setup: VideoCapture and Window creation
window = cv2.namedWindow('Human Detection', cv2.WINDOW_NORMAL)


# FPS list for each frame for avg FPS calculation

# Initialize RTDE Control
rtde_c = RTDEControl("10.132.216.193")

# Parameters
acceleration = 0.5
initial_joint_q = [-1.54, -1.83, -2.28, -0.59, 1.60, 0.023]
end_joint_q = [-0.5, -1.0, -1.0, -1.0, 1.5, 0.5]

# Speed levels based on distance ranges
speed_levels = [0.0, 0.2, 0.45, 0.7, 1.0]
max_speed = 1

# Distance thresholds (in millimeters)
distance_thresholds = [600, 1000, 1500, 2500]

# Queue for notifying speed changes
global_speed = int

def robot_motion():
    global global_speed
    current_speed_level = 1
    current_speed = speed_levels[current_speed_level] * max_speed
    try:
        while True:
            # Check if 'q' is pressed to quit
            if keyboard.is_pressed('q'):
                break

            # Check for speed updates
            try:
                new_speed_level = global_speed
                if new_speed_level != current_speed_level:
                    current_speed_level = new_speed_level
                    current_speed = speed_levels[current_speed_level] * max_speed
                    print(f"Speed updated to {current_speed}")

            except:
                pass
            
            if current_speed == 0:
                continue

            # Move to end position and back to initial position
            rtde_c.moveJ(end_joint_q, speed=current_speed, acceleration=acceleration)
            
            try:
                new_speed_level = global_speed
                if new_speed_level != current_speed_level:
                    current_speed_level = new_speed_level
                    current_speed = speed_levels[current_speed_level] * max_speed
                    print(f"Speed updated to {current_speed}")

            except:
                pass
            
            if current_speed == 0:
                continue

            rtde_c.moveJ(initial_joint_q, speed=current_speed, acceleration=acceleration)

    finally:
        rtde_c.speedStop()
        rtde_c.stopScript()
        print("Motion stopped, script terminated.")


def depth_detection():
    global global_speed
    print("depththread started")
    global depth_distance
    current_range = -1
    fps_list = []
    while True:
        start = datetime.datetime.now() # Start time for FPS calculation

        # Azure Kinect frame capture
        capture = device.update()
        ret, frame = capture.get_ir_image()
        success, depth_frame = capture.get_depth_image()
        if not ret or not success: # break if no frame
            break

        # YOLO detection
        frame, depth_distance = run_yolo(frame, depth_frame)

        # Mediapipe detection
        # frame = run_mediapipe(frame, depth_frame)

        end = datetime.datetime.now() # end time
        total = (end - start).total_seconds() # processing time

        # FPS
        fps = 1 / total 
        fps_list.append(fps)
        cv2.putText(frame, f"FPS: {fps:.2f}", (30,10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("depth", depth_frame) # show depth frame (comment out when testing

        cv2.normalize(frame, frame, 0, 65535, cv2.NORM_MINMAX)
        cv2.imshow("Human Detection", frame) # show frame (comment out when testing)

        # Determine which range the distance falls into
        if depth_distance < distance_thresholds[0]:
            new_range = 0
        elif depth_distance < distance_thresholds[1]:
            new_range = 1
        elif depth_distance < distance_thresholds[2]:
            new_range = 2
        elif depth_distance < distance_thresholds[3]:
            new_range = 3
        else:
            new_range = 4

        # If the range has changed, update the global speed
        if new_range != current_range:
            current_range = new_range
            global_speed = new_range

        if cv2.waitKey(1) == ord("q"): # force quit
            break


def process_depth(frame, depth_frame, xmin, ymin, xmax, ymax):       
    min_depth = 5000
    min_depth_x = 0
    min_depth_y = 0
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            depth = depth_frame[i][j]
            if depth < min_depth and depth != 0:
                print("depth updated")
                min_depth = depth
                min_depth_x = j
                min_depth_y = i
            
        
    cv2.circle(frame, (min_depth_x, min_depth_y), 5, GREEN, -1)

    return frame, min_depth

def visualize_YOLO(image, depth_frame, detections):
    depths = []
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), GREEN, 2)

        image, depth_distance = process_depth(image, depth_frame, xmin, ymin, xmax, ymax)
        depths.append(depth_distance)

    closest_det = min(depths) if len(depths) > 0 else 3000  
    
    return image, closest_det

def get_yolo_detections(frame):
    

    # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return model(frame, classes=0)[0] # run the model

def run_yolo(frame, depth_frame):
    frame = gray2rgb(frame)
    detections = get_yolo_detections(frame)
    frame, depth_distance= visualize_YOLO(frame, depth_frame, detections)
    return frame, depth_distance


motion_thread = threading.Thread(target=robot_motion)


motion_thread.start()

depth_detection()

motion_thread.join()