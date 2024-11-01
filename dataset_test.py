import torch
import torchvision
from femb.data.util import http_get, extract_archive
from femb.backbones import build_backbone
from femb.headers import LinearHeader, SphereFaceHeader, CosFaceHeader, ArcFaceHeader, MagFaceHeader
from femb.evaluation import VerificationEvaluator
from femb.data import LFWDataset, CelebADataset
from femb import FaceEmbeddingModel
import os

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((112, 112)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    torchvision.transforms.RandomHorizontalFlip()
    ])

# train_dataset = LFWDataset(split='train', aligned=True, transform=transform)
# val_dataset = LFWDataset(split='test', aligned=True, transform=transform)
test_path = 'datasets/lfw-deepfunneled'
people_test_path = os.path.join(test_path, 'peopleDevTest.txt')
http_get(url="http://vis-www.cs.umass.edu/lfw/peopleDevTest.txt", path=people_test_path)