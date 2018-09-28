import shutil
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchsummary.torchsummary import summary
from dataloader import VOC

import numpy as np
import matplotlib.pyplot as plt

import visdom
import yolov1

viz = visdom.Visdom()

vis_title = 'Yolo V1 Deepbaksu_vision (feat. martin, visionNoob) PyTorch on ' + 'VOC'
vis_legend = ['Train Loss']

iter_plot = yolov1.create_vis_plot(viz, 'Iteration', 'Total Loss', vis_title, vis_legend)

coord1_plot = yolov1.create_vis_plot(viz, 'Iteration', 'coord1', vis_title, vis_legend)
size1_plot = yolov1.create_vis_plot(viz, 'Iteration', 'size1', vis_title, vis_legend)

coord2_plot = yolov1.create_vis_plot(viz, 'Iteration', 'coord2', vis_title, vis_legend)
size2_plot = yolov1.create_vis_plot(viz, 'Iteration', 'size2', vis_title, vis_legend)

obj_cls_plot = yolov1.create_vis_plot(viz, 'Iteration', 'obj_cls', vis_title, vis_legend)
noobj_cls_plot = yolov1.create_vis_plot(viz, 'Iteration', 'noobj_cls', vis_title, vis_legend)

objectness1_plot = yolov1.create_vis_plot(viz, 'Iteration', 'objectness1', vis_title, vis_legend)
objectness2_plot = yolov1.create_vis_plot(viz, 'Iteration', 'objectness2', vis_title, vis_legend)


num_epochs = 40000
num_classes = 2
batch_size = 32
learning_rate = 1e-4

dropout_prop = 0.5

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# VOC Pascal Dataset
TT = "/home/keti-1080ti/Documents/dev/Yolov1/dataset/"
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

model = torch.nn.DataParallel(net, device_ids=[0, 1]).cuda()

summary(model, (3, 448,448))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):

    if (epoch == 200) or (epoch == 400) or (epoch == 600) or (epoch == 20000) or (epoch == 30000):
        scheduler.step()

    for i, (images, labels) in enumerate(train_loader):

    
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calc Loss
        loss, \
        obj_coord1_loss, \
        obj_size1_loss, \
        obj_coord2_loss, \
        obj_size2_loss, \
        obj_class_loss, \
        noobj_class_loss, \
        objness1_loss, \
        objness2_loss = yolov1.detection_loss(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:

            print('Epoch:[{}/{}], Step:[{}/{}], learning rate:{}\ttotal_loss\t{:.4f}\tcoord1\t{}\tsize1\t{}\tcoord2\t{}\tsize2\t{}\tclass\t{}\tnoobj_clss\t{}\tobjness1\t{}\tobjness2\t{}'
                  .format(epoch + 1,
                          num_epochs,
                          i + 1,
                          total_step,
                          [param_group['lr'] for param_group in optimizer.param_groups],
                          loss.item(),
                          obj_coord1_loss,
                          obj_size1_loss,
                          obj_coord2_loss,
                          obj_size2_loss,
                          obj_class_loss,
                          noobj_class_loss,
                          objness1_loss,
                          objness2_loss
                          ))
            

            yolov1.update_vis_plot(viz, (epoch+1)*batch_size +(i + 1), loss.item(), iter_plot, None, 'append')
            yolov1.update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_coord1_loss, coord1_plot, None, 'append')
            yolov1.update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_size1_loss, size1_plot, None, 'append')
            yolov1.update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_coord2_loss, coord2_plot, None, 'append')
            yolov1.update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_size2_loss, size2_plot, None, 'append')
            yolov1.update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_class_loss, obj_cls_plot, None, 'append')
            yolov1.update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), noobj_class_loss, noobj_cls_plot, None, 'append')
            yolov1.update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), objness1_loss, objectness1_plot, None, 'append')
            yolov1.update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), objness2_loss, objectness2_plot, None, 'append')


            

    if (epoch % 300) == 0:
        yolov1.save_checkpoint({
            'epoch': epoch + 1,
            'arch': "YOLOv1",
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, False, filename='checkpoint_{}.pth.tar'.format(epoch))
