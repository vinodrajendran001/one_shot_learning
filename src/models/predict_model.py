# -*- coding: utf-8 -*-
from math import inf
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import ast
from utils.siamese import Net

class NWayOneShotEvalSet(Dataset):
    """creates n-way one shot learning evaluation

    Args:
        Dataset 
    """
    def __init__(self, mainImg, imgSets, setSize, numWay, transform=None):
        self.mainImg = mainImg
        self.imgSets = imgSets
        self.setSize = setSize
        self.numWay = numWay
        self.transform = transform
    def __len__(self):
        return self.setSize
    def __getitem__(self, idx):
        # find one main image
        mainImg = Image.open(self.mainImg)
        if self.transform:
            mainImg = self.transform(mainImg)
        # process imgSet
        testSet = []
        for i in range(self.numWay):
            testImg = Image.open(self.imgSets[i])
            if self.transform:
                testImg = self.transform(testImg)
            testSet.append(testImg)
        return mainImg, testSet

if __name__ == "__main__":


    mainImg = sys.argv[1]
    imgSets = list(map(str.strip, sys.argv[2].strip('[]').split(',')))

    transformations = transforms.Compose(
        [transforms.ToTensor()]) 

    testSize = 1
    numWay = len(imgSets)

    test_set = NWayOneShotEvalSet(mainImg, imgSets, testSize, numWay, transformations)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, num_workers = 0, shuffle=False)

    #creating the original network
    save_path = 'models/siameseNet-batchnorm50.pt'  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    load_model = Net().to(device)
    state_dict = torch.load(save_path, map_location=torch.device('cpu'))
    load_model.load_state_dict(state_dict['model_state_dict'])
    load_optimizer = optim.Adam(load_model.parameters(), lr=0.0006)
    load_optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']

    print('Network loaded...', val_loss)

    with torch.no_grad():
        load_model.eval()
        print('Starting Prediction......')

        for mainimg, imgset in test_loader:
            mainimg = mainimg.to(device)
            predVal = float(-inf)
            pred = -1
            for i, testImg in enumerate(imgset):
                testImg = testImg.to(device)
                output = load_model(mainimg, testImg)
                print (i, output)
                if output > predVal:
                    pred = i
                    predVal = output

        fig = plt.figure()
        plt.title("Similar to main image")
        plt.axis('off')
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(np.asarray(Image.open(mainImg, 'r')))
        ax1.axis('off')
        ax1.title.set_text('Main image')
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(np.asarray(Image.open(imgSets[pred], 'r')))
        ax2.title.set_text('Most similar image with score ' + str(predVal))
        ax2.axis('off')
        plt.show()