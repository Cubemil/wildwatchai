import cv2
import os
import torch
from torchvision import transforms as cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import time

# Überprüfen, ob CUDA verfügbar ist und das richtige Gerät setzen
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

# Load the YOLOv5 model from torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True).to(device)
model.eval()  # Set the model to evaluation mode

# Directory containing the .MOV files
video_directory = 'C:/Users/C1/PycharmProjects/wildwatchai/LNW-01'
# Directory to save the extracted images
output_directory = 'C:/Users/C1/PycharmProjects/wildwatchai/frame-output'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define transformations
transform = T.Compose([
    T.ToTensor(),  # Convert the image to a tensor
    T.Resize((640, 640))  # Resize to 640x640
])


# Custom dataset class
class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frame_tensor


# Define a function to draw bounding boxes on the frames
def draw_bounding_boxes(frame, detections):
    for detection in detections:
        if len(detection) >= 6:
            x1, y1, x2, y2, conf, cls = detection[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


# Define a function to detect motion using optical flow
def detect_motion(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag > 2  # Threshold for motion detection


# Specify batch size
batch_size = 16

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
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps)  # Save one frame per second
            frame_number = 0
            frames = []
            prev_gray = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    motion_mask = detect_motion(prev_gray, gray)
                    if np.any(motion_mask):
                        frames.append(frame)

                prev_gray = gray if prev_gray is None else gray
                frame_number += 1

                # Process in batches
                if len(frames) == batch_size:
                    frame_dataset = FrameDataset(frames)
                    frame_loader = DataLoader(frame_dataset, batch_size=batch_size, shuffle=False)

                    start_time = time.time()
                    for batch_frames in frame_loader:
                        with torch.no_grad():
                            results = model(batch_frames.to(device))

                        for i, result in enumerate(results):
                            detections = result.cpu().numpy()
                            if len(detections) > 0:
                                frame_to_save = draw_bounding_boxes(frames[i], detections)
                                image_name = f"{os.path.splitext(filename)[0]}_frame{frame_number // frame_interval:04d}.jpg"