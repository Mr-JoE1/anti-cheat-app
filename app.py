import cv2
import dlib
import time

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load the pre-trained model for facial landmarks from the models folder
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Start capturing the video feed from the webcam
cap = cv2.VideoCapture(0)

# Variables to track the user's gaze direction and cheat status
look_away_start_time = None
look_away_threshold = 2  # seconds

def is_looking_away(landmarks):
    # Using the nose bridge and eyes landmarks to approximate gaze direction.
    # Left eye points (42-47), Right eye points (36-41), and nose bridge (27-30)
    nose_point = landmarks.part(30)
    left_eye_center = ((landmarks.part(42).x + landmarks.part(45).x) // 2,
                       (landmarks.part(42).y + landmarks.part(45).y) // 2)
    right_eye_center = ((landmarks.part(36).x + landmarks.part(39).x) // 2,
                        (landmarks.part(36).y + landmarks.part(39).y) // 2)

    # Calculate if eyes are looking away (basic approximation)
    # If the nose is too far from the center between the two eyes, the person is looking away
    if abs(nose_point.x - (left_eye_center[0] + right_eye_center[0]) // 2) > 30:
        return True
    return False

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Initialize text to display on top-left
    status_text = "Pass"
    status_color = (0, 255, 0)  # Green

    # Loop through each face detected
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Draw a green bounding box around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get facial landmarks for eyes detection
        landmarks = predictor(gray, face)

        # Check if the person is looking away
        if is_looking_away(landmarks):
            if look_away_start_time is None:
                look_away_start_time = time.time()  # Start timer when looking away
            elif time.time() - look_away_start_time > look_away_threshold:
                status_text = "Cheat Attempt"
                status_color = (0, 0, 255)  # Red
        else:
            look_away_start_time = None  # Reset the timer when looking back at the camera

        # Left eye landmarks (42-47) and Right eye landmarks (36-41)
        left_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        right_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]

        # Draw the eye bounding boxes
        for (x, y) in left_eye_points:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        for (x, y) in right_eye_points:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Display the status (Pass/Cheat Attempt) on the top-left corner
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Display the resulting frame
    cv2.imshow("Anti-Cheat App", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
