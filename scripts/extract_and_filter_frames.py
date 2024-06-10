import cv2
import os
import torch

# Load the YOLOv5 model from torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Directory containing the .MOV files
video_directory = '/Users/clemensabraham/wildwatchai/video-input'
# Directory to save the extracted images
output_directory = '/Users/clemensabraham/wildwatchai/frame-output'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define a function to check if a frame contains wildlife
def contains_wildlife(frame):
    results = model(frame)
    # Check if any detected objects are classified as wildlife
    wildlife_classes = ['deer', 'bear', 'bird', 'fox', 'rabbit', 'squirrel','feldhase', 'feldmaus','fledermaus','fuchs','reh']  # Adjust this list based on your needs
    for result in results.pred[0]:
        class_name = model.names[int(result[5])]
        if class_name in wildlife_classes:
            return True
    return False

# Walk through the directory and process each .MOV file
for root, dirs, files in os.walk(video_directory):
    for filename in files:
        if filename.endswith(".MOV"):
            video_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, video_directory)
            output_subdir = os.path.join(output_directory, relative_path)

            # Ensure the output subdirectory exists
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            cap = cv2.VideoCapture(video_path)
            count = 0
            frame_number = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Check if the frame contains wildlife
                if contains_wildlife(frame):
                    # Save the frame as an image file
                    image_name = f"{os.path.splitext(filename)[0]}_frame{frame_number:04d}.jpg"
                    image_path = os.path.join(output_subdir, image_name)
                    cv2.imwrite(image_path, frame)
                    count += 1
                frame_number += 1

            cap.release()