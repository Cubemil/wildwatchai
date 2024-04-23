# WildWatchAI

URL: https://netstorage.fb-emw.de/s/cfWSjtQsdyzkBAB <br>
PW: iEpLjFXQcH

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
    
## Model
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
