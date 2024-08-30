from rtde_control import RTDEControlInterface as RTDEControl
import keyboard
import time

# Initialize RTDE Control
rtde_c = RTDEControl("10.132.216.146")

# Parameters
acceleration = 0.5
dt = 1.0 / 500  # 2ms
initial_joint_q = [-1.54, -1.83, -2.28, -0.59, 1.60, 0.023]

# Example extended end position (adjust these angles as necessary for your robot)
end_joint_q = [-0.5, -1.0, -1.0, -1.0, 1.5, 0.5]
joint_speed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Move to initial joint position
rtde_c.moveJ(initial_joint_q)

try:
    while True:
        # Move to end position
        rtde_c.moveJ(end_joint_q)
        time.sleep(0.1)  # 1 second wait

        # Check if 'q' is pressed
        if keyboard.is_pressed('q'):
            break

        # Move back to initial position
        rtde_c.moveJ(initial_joint_q)
        time.sleep(0.1)  # 1 second wait

        # Check if 'q' is pressed
        if keyboard.is_pressed('q'):
            break
finally:
    # Ensure the robot stops if an exception occurs or 'q' is pressed
    rtde_c.speedStop()
    rtde_c.stopScript()
    print("Motion stopped, script terminated.")
