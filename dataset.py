import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.model_selection import GroupKFold
import numpy as np
import os
import cv2
import json

from PIL import Image, ImageDraw
import random

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def do_clahe(image) :
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)) 
    image[:,:,0] = clahe.apply(image[:,:,0])
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    return image

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, is_train=True, transforms=None, clahe=False, copypaste=False, k=3):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=label_root)
            for root, _dirs, files in os.walk(label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }

        print(f"Found {len(pngs)} .png files in {image_root}")
        print(f"Found {len(jsons)} .json files in {label_root}")

        if len(pngs) == 0 or len(jsons) == 0:
            raise ValueError("No .png or .json files found. Check if paths are correct and files are in the specified directories.")

        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

        assert len(jsons_fn_prefix) - len(pngs_fn_prefix) == 0, "Mismatch in .json and .png file counts"
        assert len(pngs_fn_prefix) - len(jsons_fn_prefix) == 0, "Mismatch in .png and .json file counts"

        pngs = sorted(pngs)
        jsons = sorted(jsons)
        
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        

        groups = [os.path.dirname(fname) for fname in _filenames]

        ys = [0 for fname in _filenames]

        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        dataset_no = 2
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                if i == dataset_no:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                if i != dataset_no :
                    continue
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.image_root = image_root
        self.label_root = label_root 
        
        self.clahe = clahe
        self.copypaste = copypaste
        self.k = k
    
    def __len__(self):
        return len(self.filenames)
    
    def get_coord(self, polygon):
        polygon = polygon
        for i in range(len(polygon)):
            polygon[i] = tuple(polygon[i])
        polygon_np = np.array(polygon)
        
        max = np.max(polygon_np, axis=0)
        min = np.min(polygon_np, axis=0)
        
        return max, min
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        
        if self.clahe :
            image = do_clahe(image)
            
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.copypaste:
            randoms = random.choices([i for i in range(640)], k=self.k)
            for i in randoms:
                target_image = cv2.imread(os.path.join(self.image_root, self.filenames[i])) / 255.
                target_label_path = os.path.join(self.label_root, self.labelnames[i])

                with open(target_label_path, "r") as f:
                    target_annotations = json.load(f)
                target_annotations = target_annotations["annotations"]

                for ann in target_annotations:
                    target_c = ann["label"]
                    target_c = CLASS2IND[target_c]
                    if target_c == 19 or target_c == 20 or target_c == 25 or target_c == 26:
                        points = np.array(ann["points"])
                        max, min = self.get_coord(ann['points'])
                        x = random.randint(100,1800)
                        y = random.randint(100,1800)
                        alpha = random.randint(25,50)

                        bone_area_x = [i for i in range(400,1600)]
                        while x in bone_area_x:
                            x = random.randint(100,1800)
                        x -= alpha
                        
                        img = Image.new('L', target_image.shape[:2], 0)
                        ImageDraw.Draw(img).polygon(ann['points'], outline=0, fill=1)
                        mask = np.array(img)
                        
                        new_image = cv2.bitwise_or(target_image, target_image, mask=mask)
                        image[y:y+max[1]-min[1], x:x+max[0]-min[0], ...] = new_image[min[1]:max[1], min[0]:max[0], ...]

                        ori_label = label[..., target_c]
                        ori_label[y:y+max[1]-min[1], x:x+max[0]-min[0]] = mask[min[1]:max[1], min[0]:max[0]]
                        label[..., target_c] = ori_label

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label
    
class XRayInferenceDataset(Dataset):
    def __init__(self, image_root, transforms=None, clahe=False):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
        self.image_root = image_root
        self.clahe = clahe
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        
        if self.clahe :
            image = do_clahe(image)
            
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name