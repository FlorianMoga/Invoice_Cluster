from flask import Flask
from ultralytics import YOLO
from torchvision import models, transforms
import torch
import torch.nn as nn
import math
from torchvision.transforms import functional as F


UPLOAD_FOLDER = 'static/inputs/'
RESULTS_FOLDER = 'static/outputs/'
NO_LOGO_FOLDER = r'static/clustering/no logo'

MODEL = YOLO(r"models/yolov8_logo.pt")

FEATURE_EXTRACTOR = models.densenet161()
FEATURE_EXTRACTOR.classifier = nn.Identity()
weights_path = r"models/densenet161_embedding_extractor.pt"
FEATURE_EXTRACTOR.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
FEATURE_EXTRACTOR.eval()

THRESHOLD = 0.83

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['NO_LOGO_FOLDER'] = NO_LOGO_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def add_padding(image):
    width, height = image.size

    if width > height:
        new_height = 299
        new_width = math.floor((width / height) * new_height)
    else:
        new_width = 299
        new_height = math.floor((height / width) * new_width)

    padding_width = max(0, new_width - width)
    padding_height = max(0, new_height - height)

    padding_left = padding_width // 2
    padding_right = padding_width - padding_left
    padding_top = padding_height // 2
    padding_bottom = padding_height - padding_top

    image = F.pad(image, [padding_left, padding_top, padding_right, padding_bottom], fill=0)

    return image


TRANSFORMS_DENSENET = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Lambda(lambda image: add_padding(image)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
