import cv2
import numpy as np
import face_recognition
import os

# Load the known encodings from the file
known_encodings = np.load('registered_encodings.npy')

# Create a folder to save captured frames
capture_folder = 'captured_frames'
if not os.path.exists(capture_folder):
    os.makedirs(capture_folder)

# Initialize the webcam capture
capture = cv2.VideoCapture(0)

# Check if the camera is opened
if not capture.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Function to capture 20 frames and save them
def capture_frames(num_frames):
    captured_frame_count = 0
    while captured_frame_count < num_frames:
        # Capture a frame from the webcam
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # Convert the frame to RGB (face_recognition uses RGB, OpenCV uses BGR by default)
        rgb_frame = frame[:, :, ::-1]

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            print("No faces detected in the frame.")
            continue

        # Extract face encodings for the detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Save each captured frame to the folder
        for encoding in face_encodings:
            frame_filename = os.path.join(capture_folder, f'frame_{captured_frame_count + 1}.jpg')
            cv2.imwrite(frame_filename, frame)
            captured_frame_count += 1
            print(f"Captured frame {captured_frame_count}/{num_frames}")

        # Show the captured frame with a rectangle drawn around the detected face
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the video feed
        cv2.imshow('Capture Process', frame)

        # Wait for a key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Finished capturing {num_frames} frames.")
    cv2.destroyAllWindows()
    authenticate_user()

# Function to authenticate the user based on the saved frames
def authenticate_user():
    frame_encodings = []

    # Extract face encodings from the saved frames
    for filename in os.listdir(capture_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(capture_folder, filename)
            image = cv2.imread(image_path)

            # Convert the image from BGR (OpenCV default) to RGB (for face_recognition)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect face locations in the image
            face_locations = face_recognition.face_locations(rgb_image)

            # Extract face encodings for the detected faces
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

capture_frames(20)        








