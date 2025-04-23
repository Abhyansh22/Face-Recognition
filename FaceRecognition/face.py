import cv2 as cv
import os
import time
import numpy as np
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk



# Initialize the GUI application
root = tk.Tk()
root.title("Face Detection and Capture")
root.geometry("800x600")

# Global variables
img = None
frame_count = 0
start_time = 0
last_saved_time = 0
output_folder = r'C:\Users\produ\Desktop\CIJ\python_only\OCVVP\frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set up the VideoCapture object
capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier("C:\\Users\\produ\\Desktop\\CIJ\\Code_py\\haar_face.xml")

# Function to start video capture and detection
def start_capture():
    global img, frame_count, start_time, last_saved_time
    frame_count = 0
    start_time = time.time()
    last_saved_time = time.time()
    capture_video()

# Function to stop video capture
def stop_capture():
    global img
    img = None
    capture.release()
    cv.destroyAllWindows()
    root.quit()

# Function to capture and display video frames
def capture_video():
    global img, frame_count, start_time, last_saved_time
    ret, frame = capture.read()
    if not ret:
        return

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

    if 2 > len(faces_rect) > 0 and (time.time() - last_saved_time) > 0.2:
        frame_filename = os.path.join(output_folder, f'Photoo{frame_count:03d}.jpg')
        cv.imwrite(frame_filename, frame)
        frame_count += 1
        last_saved_time = time.time()

    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    # Convert the frame to ImageTk format
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)

    # Update the canvas with the new frame
    canvas.create_image(0, 0, anchor=tk.NW, image=img)

    # Recursively call capture_video every 10 ms
    if frame_count < 30 and (time.time() - start_time) <= 10:
        root.after(10, capture_video)
    else:
        capture.release()
        cv.destroyAllWindows()
        canvas.quit()
        print('Capture complete')
        print(f"Frames saved in folder: {output_folder}")

# Create GUI elements
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

start_button = Button(root, text="Start Capture", command=start_capture)
start_button.pack(side=tk.LEFT, padx=10, pady=10)

stop_button = Button(root, text="Stop Capture", command=quit)
stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Run the application
root.mainloop()



