## WildWatchAI

# Daten
URL: https://netstorage.fb-emw.de/s/cfWSjtQsdyzkBAB <br>
PW: iEpLjFXQcH

Extrahierte Frames: https://cloud.hs-anhalt.de/s/sjH4Am2B36FZAfY

Liste zu erkennender Tierarten: https://cloud.hs-anhalt.de/s/frgPmPHoPotQrpX
    - Einige kommen recht selten vor (mit * markiert), diese sind nicht Teil des Testsets und müssen nicht erkannt werden. Falls Ihr Team diese dennoch erkennen kann, gibt es sicherlich ein paar Bonuspunkte!

## Git Links
https://github.com/google-research/vision_transformer/blob/main/vit_jax/

## Models
- DETR (End-to-End Object Detection) model with ResNet-50 backbone

DEtection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images). It was introduced in the paper End-to-End Object Detection with Transformers by Carion et al. and first released in this repository.

Disclaimer: The team releasing DETR did not write a model card for this model so this model card has been written by the Hugging Face team.
Model description

The DETR model is an encoder-decoder transformer with a convolutional backbone. Two heads are added on top of the decoder outputs in order to perform object detection: a linear layer for the class labels and a MLP (multi-layer perceptron) for the bounding boxes. The model uses so-called object queries to detect objects in an image. Each object query looks for a particular object in the image. For COCO, the number of object queries is set to 100
    -> https://huggingface.co/facebook/detr-resnet-50

- Ultralytics YOLOv8 
    -> https://huggingface.co/Ultralytics/YOLOv8

- YOLOv8n validated for Unity Sentis (Version 1.4.0-pre.3*): 
    YOLOv8n is a real-time multi-object recognition model confirmed to run in Unity 2023.
    -> https://huggingface.co/unity/sentis-YOLOv8n

# Aufgabe
 FB5 – Informatik und Sprachen: Künstliche Intelligenz (IMS)
* Wie können Methoden der KI für das Monitoring der Räuberdichte eingesetzt werden?
* Verfügbare Daten:
* Videoaufnahmen von mehreren Kameras
Zielstellung:
* Ideen für ein mögliches Projekt zusammentragen 
* Welche KI-Methoden könnten zur Videoanalyse eingesetzt werden?
* Wie könnten Ergebnisse präsentiert werden?
Aufgabenstellung:
* Entwickeln Sie ein „kleines“ Projekt zum Thema Räuberdichte.
* Es soll mind. ein KI-Modell trainiert oder angepasst werden.
* Ein Gamification-Ansatz soll (wenn möglich) erkennbar sein.

# Bibliotheken:
- Py Torch Vision: https://pytorch.org/vision/stable/index.html
- Python Referenz Trainingsscripte: https://github.com/pytorch/vision/tree/main/references/classification

# Paper:
- Review of deep learning approaches, https://thesai.org/Downloads/Volume14No11/Paper_144-A_Comprehensive_Review_of_Deep_Learning_Approaches.pdf

- Animal image identification and classification using deep neural networks techniques, https://www.sciencedirect.com/science/article/pii/S2665917422002458?via%3Dihub

- WildARe-YOLO: A lightweight and efficient wild animal recognition model, https://www.sciencedirect.com/science/article/pii/S1574954124000839

- Machine learning for inferring animal behaviour from location and movement data, https://www.sciencedirect.com/science/article/pii/S1574954118302036

- (Bayern) identification of animals and recognition of their actions in wildlife videos using deep learning techniques, https://www.sciencedirect.com/science/article/abs/pii/S1574954121000066

- object classification and visualisation with edge artificial intelligence for a customised camera trap platform, https://www.sciencedirect.com/science/article/pii/S157495412300482X

-a method for automatic identification and separation of wildlife images using ensemble learning https://www.sciencedirect.com/science/article/abs/pii/S1574954123002911


# Pretrained Models:
- Trail Camera Animal Detection, 1239 Images, https://universe.roboflow.com/sanskriti-jain/trail-camera-animal-detection/model/6
- My Datasets, 4473 Images: https://universe.roboflow.com/tayfun-kok-dvf7t/my-datasets-7tndo/dataset/7

# Image Datasets:
- Coco: 79 Kategorien
- Object365: 365 Kategorien
- lvis: 1202 Kategorien
- open-images-v7: 600 Kategorien

- Animals-detection-images-dataset: 21 Kategorien, kaggle.com https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset

- Animal-kingdom: https://paperswithcode.com/paper/animal-kingdom-a-large-and-diverse-dataset

- Ap-10k: 10kimgaes, 23 animal families, 60 species: https://paperswithcode.com/dataset/ap-10k

- Bavarian Highway Directorate, Germany 

- ImageNet

# Roadmap
## Zeit
- wöchentliche Aufgaben bzw. neue Steps
- Übung jede Woche
- 1x pro Woche Meeting (remote evtl)
- 5h Arbeitsaufwand
    - Wissenserweiterung in VL, Übung und Praktikum
    - externe Arbeit am Projekt

## Knowledge 
- basic Python Skills bei allen Mitgliedern
- Vorwissen aus maschinellem Lernen
- 

## Hardware
- GPU Power benötigt
- VCS und remotes arbeiten per gitlab repo
- PyCharm Projekte
    - benötigt Modell, Python Interpreter und durchdachte Dateistruktur
    

<br><br><br><br>
<br><br><br><br>


## Authors and acknowledgment
Developers:
- Felix G.
- Phillip J.
- Clemens A.
- Emil P.

## License
This project is being developed under supervision by Hochschule Anhalt University of Applied Sciences
