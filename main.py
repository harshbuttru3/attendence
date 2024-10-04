import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime

# Dataset path (assuming it's in the same directory as the script)
dataset_path = os.path.dirname(__file__) + '/dataset'

# Function to load known faces and roll numbers
def load_known_faces_and_rolls(dataset_path):
    known_face_encodings = []
    known_face_names = []
    known_roll_numbers = []
    
    for user_folder in os.listdir(dataset_path):
        user_folder_path = os.path.join(dataset_path, user_folder)
        roll_number = "Unknown"

        # Check if a roll_number.txt file exists in the folder
        roll_file_path = os.path.join(user_folder_path, 'reg.txt')
        if os.path.exists(roll_file_path):
            with open(roll_file_path, 'r') as roll_file:
                roll_number = roll_file.read().strip()

        for image_file in os.listdir(user_folder_path):
            image_path = os.path.join(user_folder_path, image_file)

            # Check if the file is an image
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Load the image and encode the face
                image = face_recognition.load_image_file(image_path)
                try:
                    face_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(user_folder)  # Use folder name as the person's name
                    known_roll_numbers.append(roll_number)
                except IndexError:
                    print(f"Face not found in {image_file}")
            else:
                print(f"Skipping non-image file: {image_file}")
    
    return known_face_encodings, known_face_names, known_roll_numbers

# Initialize CSV file for attendance
def initialize_csv():
    file_exists = os.path.isfile('attendance.csv')
    if not file_exists:
        with open('attendance.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Roll Number', 'Name', 'Time'])

# Function to mark attendance in the CSV file
def mark_attendance(name, roll_number):
    with open('attendance.csv', 'a', newline='') as f:
        csv_writer = csv.writer(f)
        now = datetime.now()
        time_string = now.strftime('%Y-%m-%d %H:%M:%S')
        csv_writer.writerow([roll_number, name, time_string])

# Load known face encodings, names, and roll numbers
known_face_encodings, known_face_names, known_roll_numbers = load_known_faces_and_rolls(dataset_path)

# Initialize a set to track recognized users
recognized_users = set()

# Initialize the CSV file if not already done
initialize_csv()

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Optionally, set camera resolution (e.g., for higher accuracy at a distance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Capture frame from video
    ret, frame = cap.read()

    # If frame not captured correctly, break the loop
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image from BGR (OpenCV format) to RGB (for face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Resize frame for faster processing (you can adjust this factor)
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    # Detect faces in the resized image
    face_locations = face_recognition.face_locations(small_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Loop through each face detected in the frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        roll_number = "Unknown"

        # Find the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            roll_number = known_roll_numbers[best_match_index]

        # If the name is recognized and not already marked, mark attendance
        if name not in recognized_users:
            recognized_users.add(name)
            print(f"Marking attendance for {name} (Roll Number: {roll_number})")
            mark_attendance(name, roll_number)

        # Scale back up face location since the frame was resized
        top, right, bottom, left = face_location
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f'{name} ({roll_number})', (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Display the frame with the drawings
    cv2.imshow('Attendance System', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
