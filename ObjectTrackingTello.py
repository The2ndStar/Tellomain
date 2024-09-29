from djitellopy import Tello
import cv2
import numpy as np
import time
from threading import Thread

######################################################################
# Constants
width = 600  # Width of the image
height = 400  # Height of the image
deadZone = 100  # Dead zone for direction detection
######################################################################

startCounter = 0
keepRecording = True  # Global variable to control recording state
dir = 0  # Direction variable

# Connect to Tello
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0

print("Battery Percentage:", me.get_battery())

# Start video stream
me.streamoff()
me.streamon()

# Frame dimensions
frameWidth = width
frameHeight = height

# Global variable for contours
global imgContour
imgContour = None  # Initialize as None

def videoRecorder():
    """Records video from the drone's camera."""
    frame_read = me.get_frame_read()
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter('video1.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(frame_read.frame)
        time.sleep(1 / 30)

    video.release()

# Start video recording in a separate thread
recorder = Thread(target=videoRecorder)
recorder.start()

def empty(a):
    """Empty callback for trackbars."""
    pass

# Create trackbars for HSV values
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 20, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 40, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 148, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 89, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

# Create trackbars for Canny edge detection
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 166, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 171, 255, empty)
cv2.createTrackbar("Area", "Parameters", 1750, 30000, empty)

def stackImages(scale, imgArray):
    """Stacks images for display."""
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    height, width = imgArray[0][0].shape[:2]

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        return np.vstack(hor)
    else:
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        return np.hstack(imgArray)

def getContours(img, imgContour):
    """Finds and draws contours on the image."""
    global dir
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cx = int(x + (w / 2))  # Center x of the object
            cy = int(y + (h / 2))  # Center y of the object

            # Determine direction based on object's position
            if (cx < int(frameWidth / 2) - deadZone):
                cv2.putText(imgContour, " GO LEFT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (0, int(frameHeight / 2 - deadZone)), (int(frameWidth / 2) - deadZone, int(frameHeight / 2) + deadZone), (0, 0, 255), cv2.FILLED)
                dir = 1
            elif (cx > int(frameWidth / 2) + deadZone):
                cv2.putText(imgContour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2 + deadZone), int(frameHeight / 2 - deadZone)), (frameWidth, int(frameHeight / 2) + deadZone), (0, 0, 255), cv2.FILLED)
                dir = 2
            elif (cy < int(frameHeight / 2) - deadZone):
                cv2.putText(imgContour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2 - deadZone), 0), (int(frameWidth / 2 + deadZone), int(frameHeight / 2) - deadZone), (0, 0, 255), cv2.FILLED)
                dir = 3
            elif (cy > int(frameHeight / 2) + deadZone):
                cv2.putText(imgContour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2 - deadZone), int(frameHeight / 2) + deadZone), (int(frameWidth / 2 + deadZone), frameHeight), (0, 0, 255), cv2.FILLED)
                dir = 4
            else:
                dir = 0

            cv2.line(imgContour, (int(frameWidth / 2), int(frameHeight / 2)), (cx, cy), (0, 0, 255), 3)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, " " + str(int(x)) + " " + str(int(y)), (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        else:
            dir = 0

def display(img):
    """Displays lines and circles to indicate dead zones."""
    cv2.line(img, (int(frameWidth / 2) - deadZone, 0), (int(frameWidth / 2) - deadZone, frameHeight), (255, 255, 0), 3)
    cv2.line(img, (int(frameWidth / 2) + deadZone, 0), (int(frameWidth / 2) + deadZone, frameHeight), (255, 255, 0), 3)
    cv2.circle(img, (int(frameWidth / 2), int(frameHeight / 2)), 5, (0, 0, 255), 5)
    cv2.line(img, (0, int(frameHeight / 2) - deadZone), (frameWidth, int(frameHeight / 2) - deadZone), (255, 255, 0), 3)
    cv2.line(img, (0, int(frameHeight / 2) + deadZone), (frameWidth, int(frameHeight / 2) + deadZone), (255, 255, 0), 3)
    cv2.circle(img, (int(frameWidth / 2), int(frameHeight / 2)), 5, (0, 0, 255), 5)

def move_drone():
    """Controls drone movement based on direction detected."""
    global dir
    if dir == 1:  # Move left
        me.left_right_velocity = -20
        me.for_back_velocity = 0
    elif dir == 2:  # Move right
        me.left_right_velocity = 20
        me.for_back_velocity = 0
    elif dir == 3:  # Move up
        me.for_back_velocity = 0
        me.up_down_velocity = 20
    elif dir == 4:  # Move down
        me.for_back_velocity = 0
        me.up_down_velocity = -20
    else:  # Stop
        me.left_right_velocity = 0
        me.for_back_velocity = 0
        me.up_down_velocity = 0

while True:
    # Read frame from Tello
    frame_read = me.get_frame_read()
    img = frame_read.frame
    imgContour = img.copy()

    # Convert to HSV and apply mask
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hMin = cv2.getTrackbarPos("HUE Min", "HSV")
    hMax = cv2.getTrackbarPos("HUE Max", "HSV")
    sMin = cv2.getTrackbarPos("SAT Min", "HSV")
    sMax = cv2.getTrackbarPos("SAT Max", "HSV")
    vMin = cv2.getTrackbarPos("VALUE Min", "HSV")
    vMax = cv2.getTrackbarPos("VALUE Max", "HSV")

    # Create mask based on HSV values
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    mask = cv2.inRange(imgHSV, lower, upper)

    # Find contours in the mask
    getContours(mask, imgContour)

    # Move the drone based on the detected direction
    move_drone()

    # Display images
    stacked_images = stackImages(0.6, ([img, mask], [imgContour, imgContour]))
    display(stacked_images)

    cv2.imshow("Stacked Images", stacked_images)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
keepRecording = False
recorder.join()
me.land()  # Land the drone safely
me.end()  # Disconnect from Tello
cv2.destroyAllWindows()
