import os
import cv2
import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Verzeichnisse
input_dir = 'C:\\Users\\C1\\PycharmProjects\\wildwatchai\\LNW-01'
output_dir = 'C:\\Users\\C1\\PycharmProjects\\wildwatchai\\OutputFrames'

# Erstellen des Ausgabe-Verzeichnisses, falls es nicht existiert
os.makedirs(output_dir, exist_ok=True)

# Modell laden
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transformationen definieren
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Unterstützte Videoerweiterungen (Groß- und Kleinschreibung berücksichtigen)
video_extensions = ('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')


def process_video(video_path, output_dir):
    video_name = os.path.basename(video_path)
    print(f"Verarbeite Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Bildtransformationen anwenden
        img = transform(frame)
        img = img.unsqueeze(0)

        # Inferenz
        with torch.no_grad():
            predictions = model(img)

        # Überprüfen, ob irgendwelche Objekte mit einer Vertrauensschwelle > 0.5 erkannt wurden
        scores = predictions[0]['scores'].numpy()
        if any(score > 0.5 for score in scores):
            # Frame speichern
            output_path = os.path.join(output_dir, f'{video_name}_frame_{frame_idx}.jpg')
            cv2.imwrite(output_path, frame)
            print(f"Frame {frame_idx} gespeichert: {output_path}")

        frame_idx += 1

    cap.release()
    print(f"Fertig mit Video: {video_path}")


# Durchlaufe alle Videos im Eingabeverzeichnis und Unterverzeichnissen
found_videos = False
for root, dirs, files in os.walk(input_dir):
    print(f"Durchsuche Verzeichnis: {root}")
    for video_file in files:
        if video_file.endswith(video_extensions):
            found_videos = True
            video_path = os.path.join(root, video_file)
            process_video(video_path, output_dir)

if not found_videos:
    print(f"Keine Videos im Verzeichnis gefunden: {input_dir}")

print("Videoanalyse abgeschlossen.")