import os
import pandas as pd
import cv2
import SimpleITK as sitk
from skimage import io
import json
from torchvision.transforms import ToTensor, Normalize
import numpy as np
class ImagePreprocess:
    def __init__(self, image):
        self.image = image
    def downSampling(self, target_size=(256, 256)):
        return cv2.resize(self.image, target_size, interpolation = cv2.INTER_AREA)
    
        
class CustomDataset(object):
    def __init__(self, dataframe:pd.DataFrame, image_dir:str, json_path:str, target_size:tuple|list=(224,224), normalize:bool=False):
        super(CustomDataset, self).__init__()
        self.id = [id.replace(".jpg","") for id in dataframe["image_id"]]
        self.image_dir = image_dir
        self.image_path = [os.path.join(image_dir, "Fractured",id+".jpg") if os.path.isfile(os.path.join(image_dir, "Fractured",id+".jpg")) else os.path.join(image_dir, "Non_fractured", id+".jpg") for id in self.id]
        self.bbox_path = json_path
        with open(self.bbox_path, "r") as f:
            self.json_data = json.load(f)
        self.fractured_image_names = [fracImage["file_name"] for fracImage in self.json_data["images"]]
        self.target_size= target_size
        self.normalize_mode = normalize
        self.normalize = Normalize(0.2, 0.19)
        self.to_tensor = ToTensor()
    def __getitem__(self, i):
        file_name = self.id[i]
        file_image_path = self.image_path[i]
        if file_name+".jpg" in self.fractured_image_names:
            fractured_image = io.imread(file_image_path, as_gray=True)
            imageProcess = ImagePreprocess(fractured_image)
            
            fractured_image = imageProcess.downSampling(target_size=self.target_size)
            fractured_image = self.to_tensor(fractured_image)
            if self.normalize_mode :
                fractured_image = self.normalize(fractured_image)
            
            id= self.fractured_image_names.index(file_name+".jpg")
            annotation = self.json_data["annotations"][id]
            bbox = annotation["bbox"]
            ret={
                "image": fractured_image,
                "label": 1,
                "bbox": bbox
            }
            return ret
            
        else:
            try: normal_image = io.imread(file_image_path, as_gray=True)
            except: 
                # print(file_image_path)
                normal_image = np.zeros(self.target_size)
            imageProcess = ImagePreprocess(normal_image)
            
            normal_image = imageProcess.downSampling(target_size=self.target_size)
            normal_image = self.to_tensor(normal_image)
            
            if self.normalize_mode :
                normal_image = self.normalize(normal_image)
            
            bbox=None
            ret={
                "image": normal_image,
                "label": 0,
                "bbox":None 
            }
            
            return ret
    def __add__(self, other):
        self.id = self.id+ other.id
        self.image_path = [os.path.join(self.image_dir, "Fractured",id+".jpg") if os.path.isfile(os.path.join(self.image_dir, "Fractured",id+".jpg")) else os.path.join(self.image_dir, "Non_fractured", id+".jpg") for id in self.id]
        return self
    def __len__(self):
        return len(self.id)