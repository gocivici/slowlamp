# Import the necessary libraries
import cv2

# Initialize the camera module
cap = cv2.VideoCapture(0)

# Check if the camera is initialized correctly
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Display the camera feed
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resources
cap.release()
cv2.destroyAllWindows()
