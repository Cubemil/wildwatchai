import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor
)
from pytorch_lightning.loggers import TensorBoardLogger
import supervision as sv

# Install dependencies
os.system("pip install torch")
os.system("pip install -i https://test.pypi.org/simple/ supervision==0.3.0")
os.system("pip install -q transformers")
os.system("pip install -q pytorch-lightning")
os.system("pip install -q roboflow")
os.system("pip install -q timm")
os.system("pip install pycocotools")

# Environment setup
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)

dataset_location = r"C:\Users\C1\PycharmProjects\wildwatchai\datasets\v3"
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset_location, "train")
VAL_DIRECTORY = os.path.join(dataset_location, "valid")
TEST_DIRECTORY = os.path.join(dataset_location, "test")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, image_processor, train: bool = True):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)

print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))

# Select random image
image_ids = TRAIN_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
print('Image #{}'.format(image_id))

# Load image and annotations
image = TRAIN_DATASET.coco.loadImgs(image_id)[0]
annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])
image = cv2.imread(image_path)

# Annotate
detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

# Use id2label function for training
categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k, v in categories.items()}

labels = [
    f"{id2label[class_id]}"
    for _, _, class_id, _
    in detections
]

box_annotator = sv.BoxAnnotator()
frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=16, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=16)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=16)

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.model = DetrForObjectDetection.from_pretrained(CHECKPOINT)

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        labels = batch['labels']

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        labels = batch['labels']

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        val_loss = outputs.loss
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        labels = batch['labels']

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        test_loss = outputs.loss
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

# Initialize the model
detr_model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

# Set up TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="detr_model")

# Create a PyTorch Lightning trainer with the TensorBoard logger
trainer = pl.Trainer(
    max_epochs=200,
    logger=logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1 if torch.cuda.is_available() else None
)

# Train the model
trainer.fit(detr_model, train_dataloaders=TRAIN_DATALOADER, val_dataloaders=VAL_DATALOADER)

# Evaluate the model on the test set
trainer.test(detr_model, dataloaders=TEST_DATALOADER, ckpt_path='best')

# Save the trained model
model_save_dir = r"C:\Users\C1\PycharmProjects\wildwatchai\dtrtrained"
model_filename = "wilddetr.pth"
model_path = os.path.join(model_save_dir, model_filename)
