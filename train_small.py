import sys

import warnings
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import imgaug as ia
import visdom

import yolov1_small

from PIL import Image
from torchsummary.torchsummary import summary
from utilities import dataloader
from utilities.dataloader import detection_collate
from utilities.dataloader import VOC
from utilities.utils import save_checkpoint
from utilities.utils import create_vis_plot
from utilities.utils import update_vis_plot
from utilities.augmentation import Augmenter
from yolov1_small import detection_loss_4_small_yolo
from imgaug import augmenters as iaa

warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

viz = visdom.Visdom(use_incoming_socket=False)
vis_title = 'Yolo V1 Deepbaksu_vision (feat. martin, visionNoob) PyTorch on ' + 'VOC'
vis_legend = ['Train Loss']
iter_plot = create_vis_plot(viz, 'Iteration', 'Total Loss', vis_title, vis_legend)
coord1_plot = create_vis_plot(viz, 'Iteration', 'coord1', vis_title, vis_legend)
size1_plot = create_vis_plot(viz, 'Iteration', 'size1', vis_title, vis_legend)
noobjectness1_plot = create_vis_plot(viz, 'Iteration', 'noobjectness1', vis_title, vis_legend)
objectness1_plot = create_vis_plot(viz, 'Iteration', 'objectness1', vis_title, vis_legend)
obj_cls_plot = create_vis_plot(viz, 'Iteration', 'obj_cls', vis_title, vis_legend)

#1. Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

USE_SANITY_CHECK = False
USE_AUGMENTAION = False
num_epochs = 16000
num_classes = 1
batch_size = 15
learning_rate = 1e-3
dropout_prop = 0.5

DATASET_PATH_MARTIN = "/home/martin/Desktop/5class/_class_balance/"
DATASET_PATH_JAEWON = "H:\VOC\VOC12\VOCdevkit_2\VOC2012"
SMALL_DATASET_PATH = "H:/person-300"

DATASET_PATH = SMALL_DATASET_PATH

#2. Data augmentation setting
if(USE_AUGMENTAION):
    seq = iaa.SomeOf(2,[
            iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
            iaa.Affine(
                translate_px={"x": 3, "y": 10},
                scale=(0.9, 0.9)
            ), # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            iaa.AdditiveGaussianNoise(scale=0.1*255),
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
            iaa.Affine(rotate=45),
            iaa.Sharpen(alpha=0.5)
        ])

    
else:
    seq = iaa.Sequential([])
    
composed = transforms.Compose([Augmenter(seq)])
    
#3. Load Dataset
train_dataset = VOC(root = DATASET_PATH_MARTIN, transform=composed, class_path="names/5class.names")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           collate_fn=detection_collate)

#4. Sanity Check for dataloader
if(USE_SANITY_CHECK):
    images, labels, size = iter(train_loader).next()
    images = images.to(device)
    labels = labels.to(device)
    plt.imshow(np.transpose(images[0],(1,2,0)))

#5. Load YOLOv1
net = yolov1_small.SmallYOLOv1()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = yolov1_small.SmallYOLOv1().to(device)    

#6. Sanity Check for output dimention
if(USE_SANITY_CHECK):
    #for just a image
    test_image = images[0]
    outputs = model(torch.cuda.FloatTensor(np.expand_dims(test_image,axis=0)))
    print(outputs.shape)

    #for images (batch size)
    outputs = model(torch.cuda.FloatTensor(images))
    print(outputs.shape)
    
# 7.Train the model   
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):

    if (epoch == 200) or (epoch == 400) or (epoch == 600) or (epoch == 20000) or (epoch == 30000):
        scheduler.step()

    for i, (images, labels, size) in enumerate(train_loader):

    
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calc Loss
        loss, \
        obj_coord1_loss, \
        obj_size1_loss, \
        obj_class_loss, \
        noobjness1_loss, \
        objness1_loss = detection_loss_4_small_yolo(outputs, labels)
        #objness2_loss = yolov1.detection_loss(outputs, labels)
        
        
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            
            print('Epoch ,[{}/{}] ,Step ,[{}/{}] ,lr ,{} ,total_loss ,{:.4f} ,coord1 ,{} ,size1 ,{} ,noobj_clss ,{} ,objness1 ,{} ,'
                  .format(epoch + 1,
                          num_epochs,
                          i + 1,
                          total_step,
                          [param_group['lr'] for param_group in optimizer.param_groups],
                          loss.item(),
                          obj_coord1_loss,
                          obj_size1_loss,
                          obj_class_loss,
                          noobjness1_loss,
                          objness1_loss
                          ))
            

            update_vis_plot(viz, (epoch+1)*batch_size +(i + 1), loss.item(), iter_plot, None, 'append')
            update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_coord1_loss, coord1_plot, None, 'append')
            update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_size1_loss, size1_plot, None, 'append')
            update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_class_loss, obj_cls_plot, None, 'append')
            update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), noobjness1_loss, noobjectness1_plot, None, 'append')
            update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), objness1_loss, objectness1_plot, None, 'append')


    if (epoch % 300) == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "YOLOv1",
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, filename='checkpoint_{}.pth.tar'.format(epoch))
