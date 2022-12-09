import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.io

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import cv2

from NN import *
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

model = efficientnet_b2(parameters=EfficientNet_B2_Weights.IMAGENET1K_V1)

def main():

    # check for GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    max_iters = 90
    # pick a batch size, learning rate
    batch_size = 64

    transform = EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()
    
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/desk_efficientnetb2')
    val_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_5/data/Imagenet64_val/val_data'
    
    valid_data = np.load(val_path, allow_pickle=True)
    valid_x, valid_y = valid_data['data'], valid_data['labels']
    valid_y = np.array(valid_y)

    # convert every image and label to a torch tensor
    val_xt = torch.from_numpy(valid_x)
    val_yt = torch.from_numpy(valid_y)

    # check for GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # initialize your custom dataset to be used with the dataloader
    val_dataset = TensorDataset(val_xt, val_yt)

    # create dataloader objects for each of the above datasets
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # set all internal weights to fixed
    for param in model.parameters():
        param.requires_grad = False
    net = model
    net.to(device)

    
    #==================================  Run Val accuracy pass  ================================#
    # correct = 0
    # total = 0
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for i, data in enumerate(val_loader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         inputs = inputs.to(torch.float32)
            
    #         # reshape inputs to match the val_set shape of 64x64 images
    #         inputs = inputs.view(-1, 3, 64, 64)
    #         inputs = transform(inputs)
            
    #         # calculate correct label predictions
    #         outputs = net(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         print("Running Validation Accuracy", (100 * correct / total))

    # print("Validation accuracy is", (100 * correct // total))

    #==================================  Run Val accuracy pass  ================================#
    # """
    # Here we will be testing images of a desk manually captured on a phone
    # """
    test_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_5/data/phone_test_images'
    test_video_path = '/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_5/data/phone_test_video.mp4'
    
    extract_frames_2(test_video_path, os.path.join(test_path, 'test'))
    
    testset = torchvision.datasets.ImageFolder(
                                               root=test_path,
                                               transform=transform
                                              )
    test_loader = DataLoader(testset, batch_size=16,
                                shuffle=True, num_workers=4, pin_memory=True)

    correct_acc = 0
    total_acc = 0
    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            total_acc += labels.size(0)
            correct_acc += (predictions == labels).sum().item()

    # print overall accuracy
    print(f'Accuracy on custom phone camera images: {100 * correct_acc // total_acc} %')

def extract_frames_2(path, dump):
    currentframe = 0

    # Read the video from specified path
    cam = cv2.VideoCapture(path)
    
    while(True):
        
        # reading from frame
        success,frame = cam.read()

        currentframe += 1
        if success and currentframe%1 == 0 :
            # if video is still left continue creating images
            frame = frame[:,150:502,:]
            image = cv2.rotate(frame, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(dump, (str(currentframe) + ".jpg")),image)
            
        if currentframe == 1000:
            break
    
    # Release all space and windows once done
    cam.release()
    

if __name__ == '__main__':
    main()