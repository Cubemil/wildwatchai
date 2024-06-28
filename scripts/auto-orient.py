import os
from moviepy.editor import VideoFileClip

def rotate_videos(input_dir, output_dir):
    # Erstellen des Ausgabe-Verzeichnisses, falls es nicht existiert
    os.makedirs(output_dir, exist_ok=True)
    print(f'Ausgabeverzeichnis: {output_dir}')

    # Alle Dateien im Eingabeverzeichnis auflisten
    files = os.listdir(input_dir)
    print(f'Gefundene Dateien im Eingabeverzeichnis: {files}')  # Debugging-Ausgabe

    # Finde alle .mov Dateien (unabhängig von Groß-/Kleinschreibung)
    mov_files = [f for f in files if f.lower().endswith('.mov')]
    print(f'Gefundene .mov Dateien: {mov_files}')  # Debugging-Ausgabe

    for mov_file in mov_files:
        try:
            # Vollständiger Pfad zur Quelldatei
            src_file = os.path.join(input_dir, mov_file)
            print(f'Verarbeite Datei: {src_file}')  # Debugging-Ausgabe

            # Vollständiger Pfad zur Zieldatei
            dst_file = os.path.join(output_dir, mov_file)
            print(f'Ausgabedatei: {dst_file}')  # Debugging-Ausgabe

            # Video laden
            video = VideoFileClip(src_file)
            print(f'Video {src_file} geladen, Dauer: {video.duration} Sekunden')  # Debugging-Ausgabe

            # Video um 180 Grad drehen
            rotated_video = video.rotate(180)
            print(f'Video {src_file} wurde um 180 Grad gedreht')  # Debugging-Ausgabe

            # Gedrehtes Video speichern
            rotated_video.write_videofile(dst_file, codec='libx264')
            print(f'{mov_file} wurde erfolgreich um 180 Grad gedreht und gespeichert.')  # Erfolgsmeldung

        except Exception as e:
            print(f'Fehler beim Verarbeiten von {mov_file}: {e}')  # Fehlerausgabe

        finally:
            # Ressourcen freigeben
            video.close()

# Verzeichnisse
input_dir = '/Users/clemensabraham/PycharmProjects/wildwatchai/LNW-02/mov'  # Pfad zum Eingabeverzeichnis
output_dir = '/Users/clemensabraham/PycharmProjects/wildwatchai/LNW-02/movorient'  # Pfad zum Ausgabe-Verzeichnis

# Funktion aufrufen
rotate_videos(input_dir, output_dir)