import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Read the video file
video_path = 'datab/IMG_2506.mp4'
cap = cv2.VideoCapture(video_path)

# List to store the angles
elbow_angles = []

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect the pose
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        # Get landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Retrieve the required landmarks for shoulder, elbow, and wrist
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate the angle
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # Append the angle to the list
        elbow_angles.append(angle)
        
        # Display the angle
        cv2.putText(frame, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )
    
    cv2.imshow('Elbow Angle', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Now `elbow_angles` contains all the angles calculated through the video
print(elbow_angles)
