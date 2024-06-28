import os
import cv2
import numpy as np
from PIL import Image

# Verzeichnisse
input_dir = '/Users/clemensabraham/PycharmProjects/wildwatchai/lnw03/sortiert'  # Ordner mit Videos
output_dir = '/Users/clemensabraham/PycharmProjects/wildwatchai/outputlnw03+02b'  # Ordner für extrahierte Bilder

# Erstellen des Ausgabe-Verzeichnisses, falls es nicht existiert
os.makedirs(output_dir, exist_ok=True)

# Unterstützte Videoerweiterungen
video_extensions = ('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')

def normalize_image(image):
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return Image.fromarray((image_array * 255).astype(np.uint8))

def extract_frames(video_path, output_dir):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames pro Sekunde
    frame_interval = int(fps)  # Intervall für einen Frame pro Sekunde
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Einen Frame pro Sekunde extrahieren
        if frame_idx % frame_interval == 0:
            # Bild in RGB konvertieren
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

            # Bild normalisieren
            image = normalize_image(image)

            # Bild speichern
            output_path = os.path.join(output_dir, f'{video_name}_frame_{frame_idx}.jpg')
            image.save(output_path, format='JPEG')
            print(f'Frame {frame_idx} gespeichert: {output_path}')

        frame_idx += 1

    cap.release()
    print(f'Fertig mit Video: {video_path}')

# Durchlaufe alle Videos im Eingabeverzeichnis und Unterverzeichnissen
found_videos = False
for root, dirs, files in os.walk(input_dir):
    print(f'Durchsuche Verzeichnis: {root}')  # Debug-Ausgabe
    for video_file in files:
        print(f'Gefundene Datei: {video_file}')  # Debug-Ausgabe
        if video_file.endswith(video_extensions):
            found_videos = True
            video_path = os.path.join(root, video_file)
            print(f'Video gefunden: {video_path}')  # Debug-Ausgabe
            extract_frames(video_path, output_dir)

if not found_videos:
    print(f'Keine Videos im Verzeichnis gefunden: {input_dir}')

print('Bildextraktion abgeschlossen.')