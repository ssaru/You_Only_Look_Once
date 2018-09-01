import shutil
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchsummary.torchsummary import summary
from dataloader import VOC

import numpy as np
import matplotlib.pyplot as plt

import yolov1

num_epochs = 16000
num_classes = 21
batch_size = 8
learning_rate = 1e-4

dropout_prop = 0.5

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# VOC Pascal Dataset
TT = "/media/martin/keti_martin/Martin/DataSet/VOC_Pascal/VOC/VOCdevkit/VOC2012"
DATASET_PATH_MARTIN = "/media/keti-ai/AI_HARD3/DataSets/VOC_Pascal/VOC/VOCdevkit/VOC2012"
DATASET_PATH_JAEWON = "D:\dataset\VOC2012"
train_dataset = VOC(root = TT,
                    transform=transforms.ToTensor(), cls_option = False, selective_cls=None)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           collate_fn=yolov1.detection_collate)


net = yolov1.YOLOv1()
# visualize_weights_distribution(net)

model = torch.nn.DataParallel(net, device_ids=[0]).cuda()

summary(model, (3, 448,448))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calc Loss
        loss = yolov1.detection_loss(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, learning rate: {}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          [param_group['lr'] for param_group in optimizer.param_groups]))

    if(epoch == 100) or (epoch == 500) or (epoch == 1000) or (epoch == 2000) or (epoch == 4000) or (epoch == 8000) or (epoch == 14000):
        scheduler.step()
            

    if (epoch % 300) == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': "YOLOv1",
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, filename='checkpoint_{}.pth.tar'.format(epoch))