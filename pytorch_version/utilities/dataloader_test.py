import torch
import torchvision.transforms as transforms

from dataloader import VOC
from dataloader_sample import DataLoader

import numpy as np

def detection_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])

        np_label = np.zeros((7,7,6), dtype=np.float32)
        for i in range(7):
            for j in range(7):
                np_label[i][j][1] = 20

        for object in sample[1]:
            objectness=1
            cls = object[0]
            x_ratio = object[1]
            y_ratio = object[2]
            w_ratio = object[3]
            h_ratio = object[4]

            # can be acuqire grid (x,y) index when divide (1/S) of x_ratio
            grid_x_index = int(x_ratio // (1/7))
            grid_y_index = int(y_ratio // (1/7))
            x_offset = x_ratio - ((grid_x_index) * (1/7))
            y_offset = y_ratio - ((grid_y_index) * (1/7))

            # insert object row in specific label tensor index as (x,y)
            # object row follow as
            # [objectness, class, x offset, y offset, width ratio, height ratio]
            np_label[grid_x_index-1][grid_y_index-1] = np.array([objectness, cls, x_offset, y_offset, w_ratio, h_ratio])

        label = torch.from_numpy(np_label)
        targets.append(label)

    return torch.stack(imgs,0), torch.stack(targets, 0)

"""
batch_size = 128
root ="/media/keti-1080ti/ketiCar/DataSet/VOC/VOCdevkit/VOC2012/"
train_dataset = VOC(root=root, transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=detection_collate)

total_step = len(train_loader)
#print("TOTL DATASET LENGTH : {}".format(total_step))


for i, (images, lables) in enumerate(train_loader):
    print(i)
    print(lables)
    print(images)
    break


"""