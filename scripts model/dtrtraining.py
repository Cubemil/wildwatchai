import os
import torch
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import psutil
import GPUtil


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"RAM Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        print(f"GPU Memory Usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, processor_name: str, annotation_file_name: str,
                 image_size=(640, 640)):
        annotation_file_path = os.path.join(image_directory_path, annotation_file_name)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = DetrImageProcessor.from_pretrained(processor_name)
        self.image_size = image_size

    def __getitem__(self, idx):
        try:
            images, annotations = super(CocoDetection, self).__getitem__(idx)
            images = images.resize(self.image_size)
            image_id = self.ids[idx]
            annotations = {'image_id': image_id, 'annotations': annotations}
            encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
            return encoding["pixel_values"].squeeze(), encoding["labels"][0]
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return None

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        pixel_values = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        return {'pixel_values': pixel_values, 'labels': labels}


class Detr(pl.LightningModule):
    def __init__(self, lr=1e-5):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        self.lr = lr

    def training_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch['pixel_values'], labels=batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch['pixel_values'], labels=batch['labels'])
        val_loss = outputs.loss
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


def load_trained_model(model_path):
    model = Detr()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    print(f"Loaded trained model from {model_path}")
    return model


def main():
    print("Initial memory usage:")
    print_memory_usage()

    CHECKPOINT = 'facebook/detr-resnet-50'
    BATCH_SIZE = 12
    NUM_WORKERS = 12
    EPOCHS = 100

    dataset_location = r"C:\Users\C1\Desktop\kilnw\dataset"
    ANNOTATION_FILE_NAME = "_annotations.coco.json"
    TRAIN_DIRECTORY = os.path.join(dataset_location, "train")
    VAL_DIRECTORY = os.path.join(dataset_location, "valid")

    train_dataset = CocoDetection(TRAIN_DIRECTORY, CHECKPOINT, ANNOTATION_FILE_NAME, image_size=(640, 640))
    val_dataset = CocoDetection(VAL_DIRECTORY, CHECKPOINT, ANNOTATION_FILE_NAME, image_size=(640, 640))

    print("After dataset creation:")
    print_memory_usage()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=CocoDetection.collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=CocoDetection.collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # Laden des vortrainierten Modells
    model_save_dir = r"C:\Users\C1\Desktop\kilnw\dtrtrained"
    model_path = os.path.join(model_save_dir, "wilddetr.pth")

    if os.path.exists(model_path):
        model = load_trained_model(model_path)
    else:
        print("No pre-trained model found. Starting from scratch.")
        model = Detr()

    # Auskommentierte Version zum Laden des ursprünglichen Modells
    # model = Detr()

    logger = TensorBoardLogger("tb_logs", name="detr_model")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu',
        devices=1,
        logger=logger,
        num_sanity_val_steps=0,
        precision=16,
        accumulate_grad_batches=4
    )

    print("Before training:")
    print_memory_usage()

    trainer.fit(model, train_loader, val_loader)

    print("After training:")
    print_memory_usage()

    # Speichern des trainierten Modells
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Final memory usage:")
    print_memory_usage()


if __name__ == '__main__':
    main()