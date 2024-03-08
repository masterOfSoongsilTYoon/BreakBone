import os
import argparse
import pandas as pd
import cv2
from skimage import io
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn
import utils
from Networks import VGG16
import numpy as np
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score


def train(net: nn.Module, epochs:int ,train_loader: DataLoader, valid_loader: DataLoader, lossf:nn.Module ,optimizer: nn.Module, DEVICE):
    net.to(DEVICE)
    net.train()
    history1 = {
        "loss": []
    }
    history2 = {
        "loss": []
    }
    for epoch in range(epochs):
        print(f"epoch : {epoch+1}")
        losses = 0
        for i, samples in enumerate(train_loader):
            images = torch.stack([sample["image"] for sample in samples], dim=0).type(torch.float32).to(DEVICE)
            labels = torch.stack([torch.tensor(sample["label"]) for sample in samples], dim=0).type(torch.float32).to(DEVICE)
            
            out = net(images)
            
            loss = lossf(out, labels.unsqueeze(dim=1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses += loss.item()
            
        avg_loss = losses/(i+1)
        print(f"train loss: {avg_loss}")
        history1["loss"].append(avg_loss)
        vhistory = valid(net, valid_loader, lossf, Device=DEVICE)
        for k, v in vhistory.items():
            history2[k].append(v)
    
    return {"train":history1, "valid":history2}    
    

def valid(net: nn.Module, valid_loader: DataLoader, lossf:nn.Module , Device):
    return {}

def parser(description="Central Train"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--version", metavar="N", type=str, default= "default", help="version name")
    parser.add_argument("--image_dir", metavar="N", type=str, help="images directory path")
    parser.add_argument("--json_path", metavar="N", type=str, help="meta json file path")
    parser.add_argument("--target_size", metavar="N", type=tuple, default=(224, 224),help="target size for train")
    parser.add_argument("--normalize", metavar="N", type=bool, default=False,help="normalize mode True or False")
    parser.add_argument("--epoch", metavar="N", type=int, default=200,help="epoch number")
    return parser.parse_args()

def make_version_directory(dir, version):
    path=os.path.join(dir, version)
    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory is done")
        
def informing_args(arg):
    print("=========================")
    print(f"version : {arg.version}")
    print(f"image_dir : {arg.image_dir}")
    print(f"json_path : {arg.json_path}")
    print(f"target_size : {str(arg.target_size)}")
    print(f"normalize mode : {str(arg.normalize)}")
    print("=========================")
    
def informing_dataset(train,test,valid):
    print("=========================")
    print(f"train data: {len(train)}")
    print(f"test data: {len(test)}")
    print(f"valid data: {len(valid)}")
    print("=========================")
    
def main():
    arg = parser()
    informing_args(arg)
    
    make_version_directory(dir="./Models", version = arg.version)
    
    fractured_train_dataframe = pd.read_csv("./Dataset/FracAtlas/Utilities/Fracture Split/train.csv")
    fractured_train_dataset = utils.CustomDataset(fractured_train_dataframe, arg.image_dir, arg.json_path, arg.target_size, arg.normalize)
    nonfractured_train_datafrme = pd.read_csv("./Dataset/FracAtlas/Utilities/Nonfractured/All.csv")
    nonfractured_train_dataset = utils.CustomDataset(nonfractured_train_datafrme, arg.image_dir, arg.json_path, arg.target_size, arg.normalize)
    
    train_dataset = fractured_train_dataset+nonfractured_train_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x:x)
    
    valid_dataframe = pd.read_csv("./Dataset/FracAtlas/Utilities/Fracture Split/valid.csv")
    valid_dataset = utils.CustomDataset(valid_dataframe, arg.image_dir, arg.json_path, arg.target_size, arg.normalize)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=lambda x:x)
    
    test_dataframe = pd.read_csv("./Dataset/FracAtlas/Utilities/Fracture Split/valid.csv")
    test_dataset = utils.CustomDataset(test_dataframe, arg.image_dir, arg.json_path, arg.target_size, arg.normalize)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x:x)
    
    informing_dataset(train_dataset, test_dataset, valid_dataset)
    
    net = VGG16()
    net.float()
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), 1e-4)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    history=train(net=net, epochs=arg.epoch, train_loader=train_dataloader, valid_loader= valid_dataloader, lossf= criterion, optimizer= optimizer, DEVICE=DEVICE)
    
if __name__=="__main__":
    main()
    