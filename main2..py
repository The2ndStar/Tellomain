import cv2
import numpy as np
from djitellopy import Tello
import time  # For timing purposes

# Initialize drone
me = Tello()
me.connect()

# Optionally, set streaming from the drone's camera
me.streamon()

# Track whether the drone has taken off already to prevent repeated takeoffs
taken_off = False
last_seen_black = time.time()  # Track the last time black was seen

while True:
    # Get the drone camera feed
    frame = me.get_frame_read().frame
    imgContour = frame.copy()

    # Convert the image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define black color range in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # Create a mask to detect black color
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours for the detected black color
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found (black object detected)
    if len(contours) > 0:
        # Draw contours on the image for visualization
        cv2.drawContours(imgContour, contours, -1, (0, 255, 0), 3)

        # Optional: display a message
        cv2.putText(imgContour, "BLACK OBJECT DETECTED", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

        # Check if the drone hasn't taken off yet
        if not taken_off:
            me.takeoff()  # Take off the drone
            taken_off = True  # Set the flag to avoid multiple takeoffs

        # Update the time black was last detected
        last_seen_black = time.time()

    # Check if black object hasn't been detected for more than 5 seconds
    if taken_off and time.time() - last_seen_black > 5:
        me.land()  # Land the drone after 5 seconds of no black detection
        break  # Exit the loop

    # Show the frame
    cv2.imshow("Contour", imgContour)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()  # Land the drone before exiting
        break

# Release all resources
cv2.destroyAllWindows()

