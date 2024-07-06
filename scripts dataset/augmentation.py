### albumentations von pytorch um selber bilder zu augmentieren

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from tqdm import tqdm
from pycocotools.coco import COCO
import json


def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(height=800, width=800, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        A.Rotate(limit=10, p=0.5),
        A.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        A.GaussianBlur(p=0.3, sigma_limit=(0.1, 2.0)),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))


def augment_dataset(input_dir, input_annotation, output_dir, output_annotation):
    coco = COCO(input_annotation)
    aug = get_augmentation()

    os.makedirs(output_dir, exist_ok=True)

    new_annotations = []
    new_images = []
    new_id = 1

    for img_id in tqdm(coco.imgs):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(input_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        bboxes = [ann['bbox'] for ann in anns]
        category_ids = [ann['category_id'] for ann in anns]

        augmented = aug(image=img, bboxes=bboxes, category_ids=category_ids)

        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_category_ids = augmented['category_ids']

        new_img_filename = f"aug_{new_id}.jpg"
        cv2.imwrite(os.path.join(output_dir, new_img_filename), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        new_images.append({
            "id": new_id,
            "file_name": new_img_filename,
            "height": aug_img.shape[0],
            "width": aug_img.shape[1]
        })

        for bbox, cat_id in zip(aug_bboxes, aug_category_ids):
            new_annotations.append({
                "id": len(new_annotations) + 1,
                "image_id": new_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })

        new_id += 1

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.dataset['categories']
    }

    with open(output_annotation, 'w') as f:
        json.dump(new_coco, f)


if __name__ == "__main__":
    input_dir = "path/to/input/images"
    input_annotation = "path/to/input/annotations.json"
    output_dir = "path/to/output/augmented/images"
    output_annotation = "path/to/output/augmented/annotations.json"

    augment_dataset(input_dir, input_annotation, output_dir, output_annotation)