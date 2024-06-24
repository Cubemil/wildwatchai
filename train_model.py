import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import detr_resnet50
from pycocotools.coco import COCO
import torchvision


class SimpleCocoDataset(Dataset):
    def __init__(self, root, annotation):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img = torchvision.io.read_image(f"{self.root}/{img_info['file_name']}")

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        return img, target

    def __len__(self):
        return len(self.ids))

def get_data_loader(root, annotation, batch_size, shuffle=True):
    dataset = SimpleCocoDataset(root, annotation)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                      collate_fn=lambda x: tuple(zip(*x)))


def get_model(num_classes):


# ... (wie zuvor)

def train_one_epoch(model, criterion, optimizer, data_loader, device):


# ... (wie zuvor)

def evaluate(model, data_loader, device):


# ... (wie zuvor)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datenlader für Training, Validierung und Test
    train_loader = get_data_loader('C:\Users\C1\PycharmProjects\wildwatchai\Animal Trail Cam.notAugmented.i.coco\train', 'C:\Users\C1\PycharmProjects\wildwatchai\Animal Trail Cam.notAugmented.i.coco\train\_annotations.json', batch_size=2)
    val_loader = get_data_loader('C:\Users\C1\PycharmProjects\wildwatchai\Animal Trail Cam.notAugmented.i.coco\valid', 'C:\Users\C1\PycharmProjects\wildwatchai\Animal Trail Cam.notAugmented.i.coco\valid\_annotations.json', batch_size=2, shuffle=False)
    test_loader = get_data_loader('C:\Users\C1\PycharmProjects\wildwatchai\Animal Trail Cam.notAugmented.i.coco\test', 'C:\Users\C1\PycharmProjects\wildwatchai\Animal Trail Cam.notAugmented.i.coco\test\_annotations.json', batch_size=2, shuffle=False)

    num_classes = 91  # COCO has 80 classes + background
    model = get_model(num_classes).to(device)

    matcher = HungarianMatcher()
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)
    criterion.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    num_epochs = 10
    best_val_map = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        print(f"Train Loss: {train_loss:.4f}")

        print("Evaluating on validation set...")
        val_evaluator = evaluate(model, val_loader, device)
        val_map = val_evaluator.coco_eval['bbox'].stats[0]  # mAP @ IoU=0.50:0.95
        print(f"Validation mAP: {val_map:.4f}")

        # Speichern des besten Modells basierend auf Validierungs-mAP
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), 'best_detr_resnet.pth')
            print("Saved new best model.")

    # Laden des besten Modells für den finalen Test
    model.load_state_dict(torch.load('best_detr_resnet.pth'))

    print("Evaluating on test set...")
    test_evaluator = evaluate(model, test_loader, device)
    test_map = test_evaluator.coco_eval['bbox'].stats[0]  # mAP @ IoU=0.50:0.95
    print(f"Test mAP: {test_map:.4f}")


if __name__ == "__main__":
    main()