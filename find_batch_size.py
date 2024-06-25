import torch
import sys
import os

# Verwenden Sie einen rohen String für den Pfad
detr_path = r"C:\Users\C1\PycharmProjects\wildwatchai\detr-resnet-50"
sys.path.append(detr_path)


# Importieren Sie die notwendigen Module aus dem geklonten Repository
from models import build_model
from util.misc import NestedTensor


def get_detr_model(num_classes):
    # Erstellen Sie ein DETR-Modell mit den Standardeinstellungen
    args = type('Args', (), {
        "num_classes": num_classes,
        "hidden_dim": 256,
        "nheads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "backbone": "resnet50",
        "dilation": False,
        "aux_loss": True,
    })()
    model, _ = build_model(args)
    return model


def find_optimal_batch_size(model, device):
    batch_size = 1
    while True:
        try:
            # DETR erwartet NestedTensor als Eingabe
            dummy_input = NestedTensor(torch.randn(batch_size, 3, 800, 800).to(device),
                                       torch.ones(batch_size, 800, 800).bool().to(device))

            model.train()
            _ = model(dummy_input)

            print(f"Batch size {batch_size} successful")
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Reached maximum batch size at {batch_size // 2}")
                return batch_size // 2
            else:
                raise e


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_classes = 91  # COCO hat 80 Klassen + 1 für den Hintergrund
    model = get_detr_model(num_classes).to(device)

    optimal_batch_size = find_optimal_batch_size(model, device)
    print(f"Optimal batch size: {optimal_batch_size}")