### automatically find moving objects in videos and save the frames as images
    ## war ungenau, hatte nicht alle Klassen
    ## am Ende wurde alles selber sortiert

import os
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.io import read_video
from torchvision.transforms.functional import to_tensor
from PIL import Image

# Verzeichnisse
input_dir = 'C:/Users/C1/PycharmProjects/wildwatchai/LNW-01'
output_dir = 'C:/Users/C1/PycharmProjects/wildwatchai/frame-output'

# Erstellen des Ausgabe-Verzeichnisses, falls es nicht existiert
os.makedirs(output_dir, exist_ok=True)

# Laden des vortrainierten Modells
model = fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Transformations
transform = T.Compose([
    T.Resize((480, 640)),
    T.ToTensor()
])

# Definieren der Klassen, die wir als Tiere betrachten
animal_classes = []  # Beispiele für COCO-Klassen (Katzen, Hunde, Pferde, etc.)


def process_video(video_path, output_dir):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konvertierung des Frames in ein PIL-Bild und Transformation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Inferenz
        with torch.no_grad():
            outputs = model(img_tensor)

        # Überprüfen, ob Tiere im Frame erkannt wurden
        for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
            if label.item() in animal_classes and score.item() > 0.5:
                # Frame speichern
                output_path = os.path.join(output_dir, f'{video_name}_frame_{frame_idx}.jpg')
                cv2.imwrite(output_path, frame)
                break

        frame_idx += 1

    cap.release()


# Durchlaufe alle Videos im Eingabeverzeichnis
for video_file in os.listdir(input_dir):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(input_dir, video_file)
        process_video(video_path, output_dir)

print("Videoanalyse abgeschlossen.")