import os
import shutil

def find_and_copy_pairs(input_dir, pairs_dir, nulled_dir):
    # Erstellen der Ausgabe-Verzeichnisse, falls sie nicht existieren
    os.makedirs(pairs_dir, exist_ok=True)
    os.makedirs(nulled_dir, exist_ok=True)

    # Alle Dateien im Eingabeverzeichnis auflisten
    files = os.listdir(input_dir)

    # Erstelle Sets von .jpg und .xml Dateien
    jpg_files = set(f for f in files if f.endswith('.jpg'))
    xml_files = set(f for f in files if f.endswith('.xml'))

    # Kopiere gleichnamige Paare und JPG-Dateien ohne entsprechende XML-Datei
    for jpg_file in jpg_files:
        xml_file = os.path.splitext(jpg_file)[0] + '.xml'
        src_jpg = os.path.join(input_dir, jpg_file)

        if xml_file in xml_files:
            # Vollständige Pfade für die Zieldateien in pairs
            src_xml = os.path.join(input_dir, xml_file)
            dst_jpg = os.path.join(pairs_dir, jpg_file)
            dst_xml = os.path.join(pairs_dir, xml_file)

            # Kopiere die Dateien in das pairs-Verzeichnis
            shutil.copy(src_jpg, dst_jpg)
            shutil.copy(src_xml, dst_xml)
            print(f'Kopiert: {jpg_file} und {xml_file} in pairs')
        else:
            # Vollständige Pfade für die Zieldateien in nulled
            dst_jpg = os.path.join(nulled_dir, jpg_file)

            # Kopiere die .jpg Datei in das nulled-Verzeichnis
            shutil.copy(src_jpg, dst_jpg)
            print(f'Kein entsprechendes XML für {jpg_file} gefunden. Datei in Nulled kopiert.')

# Verzeichnisse
input_dir = '/Users/clemensabraham/PycharmProjects/wildwatchai/LostAnimals/frames'  # Pfad zum Eingabeverzeichnis
pairs_dir = '/Users/clemensabraham/PycharmProjects/wildwatchai/pairs'  # Pfad zum pairs-Verzeichnis
nulled_dir = '/Users/clemensabraham/PycharmProjects/wildwatchai/nulled'  # Pfad zum nulled-Verzeichnis

# Funktion aufrufen
find_and_copy_pairs(input_dir, pairs_dir, nulled_dir)