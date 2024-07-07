## WildWatchAI

## Introduction

This repository contains the code and documentation for fine-tuning the facebook detr-resnet-50 model from huggingface.com on our custom dataset on local nvidia 4080 gpu. This project demonstrates the process of annotating a custom dataset, training the detr-resnet model and evaluating its performance.


## Dataset

The dataset consists of self-annotated images of various animals. The images were annotated by hand with bounding boxes. The images were augmented and split 70/20/10 into Train/Validation/Test subsets and evenly distributed using RoboFlow's web interface. The annotations were exported in the COCO format, which includes the class label and the bounding box coordinates in a .json file.


# Dataset source:
URL: https://netstorage.fb-emw.de/s/cfWSjtQsdyzkBAB <br>
PW: iEpLjFXQcH

# Annotation tool:
https://www.makesense.ai

# Data preperation scripts (scripts dataset folder):
- Extract frames from video files: extract-frames.py
- Move extracted frames to orient: auto-orient.py
- Sort annotaded frames: SortXmljpeg.py
- Custom frame augmentation: augmentatiopn.py

# Training dataset:
https://app.roboflow.com/wildwatch/animal-trail-detection/2
or
https://cloud.hs-anhalt.de/s/W2b7QMC9TgMcpqr

# Extrahierte Frames: https://cloud.hs-anhalt.de/s/sjH4Am2B36FZAfY

Liste zu erkennender Tierarten: https://cloud.hs-anhalt.de/s/frgPmPHoPotQrpX
    - Einige kommen recht selten vor (mit * markiert), diese sind nicht Teil des Testsets und müssen nicht erkannt werden. Falls Ihr Team diese dennoch erkennen kann, gibt es sicherlich ein paar Bonuspunkte!


## Model Training

1. **Prepared Dataset**: https://cloud.hs-anhalt.de/s/W2b7QMC9TgMcpqr Annotated, augmented, polished, split, and exported data.
2. **Installed Dependencies in virtual environment**: requirements.txt installed the necessary libraries and dependencies, including PyTorch, pycocotools, transformers, supervision, roboflow etc.
3. **Train the Model**: Clone git repository and check correct file path destinations. Run the detrtraining.py script (in folder: script training) to train the model. Epochs: ~250

- write in console to run tensorboard: --logdir=C:\Users\C1\pycharmprojects\wildwatchai\tb_logs

- finetuning gpu setup (find batch size for gpu): find_batch_size.py 


## Model introduction
- DETR (End-to-End Object Detection) model with ResNet-50 backbone

DEtection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images). It was introduced in the paper End-to-End Object Detection with Transformers by Carion et al. and first released in this repository.

Disclaimer: The team releasing DETR did not write a model card for this model so this model card has been written by the Hugging Face team.
Model description

The DETR model is an encoder-decoder transformer with a convolutional backbone. Two heads are added on top of the decoder outputs in order to perform object detection: a linear layer for the class labels and a MLP (multi-layer perceptron) for the bounding boxes. The model uses so-called object queries to detect objects in an image. Each object query looks for a particular object in the image. For COCO, the number of object queries is set to 100
    -> https://huggingface.co/facebook/detr-resnet-50


## Aufgabe
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

- a method for automatic identification and separation of wildlife images using ensemble learning https://www.sciencedirect.com/science/article/abs/pii/S1574954123002911


# Pretrained Models:
- Trail Camera Animal Detection, 1239 Images, https://universe.roboflow.com/sanskriti-jain/trail-camera-animal-detection/model/6
- My Datasets, 4473 Images: https://universe.roboflow.com/tayfun-kok-dvf7t/my-datasets-7tndo/dataset/7


## Roadmap
# Zeit
- wöchentliche Aufgaben bzw. neue Steps
- Übung jede Woche
- 1x pro Woche Meeting (remote evtl)
- 5h Arbeitsaufwand
    - Wissenserweiterung in VL, Übung und Praktikum
    - externe Arbeit am Projekt


# Hardware
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
