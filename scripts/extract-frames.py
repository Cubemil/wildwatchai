import cv2  # OpenCV library for image and video processing
import os   # OS library for file and directory operations

# Directory containing the .MOV files
video_directory = '/Users/clemensabraham/wildwatchai/video-input/*'
# Directory to save the extracted images
output_directory = '/Users/clemensabraham/wildwatchai/frame-output/'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over video files in the directory
for filename in os.listdir(video_directory):
    if filename.endswith(".MOV"):
        video_path = os.path.join(video_directory, filename)
        cap = cv2.VideoCapture(video_path)  # Capture the video
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()  # Read a frame
            if not ret:
                break  # Exit if no frame is returned
            # Save the frame as an image file
            image_name = f"{os.path.splitext(filename)[0]}_frame{count:04d}.jpg"
            image_path = os.path.join(output_directory, image_name)
            cv2.imwrite(image_path, frame)
            count += 1

        cap.release()  # Release the video capture object