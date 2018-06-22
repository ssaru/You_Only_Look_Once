import torch
import torchvision.transforms as transforms

from dataloader import VOC
from dataloader_sample import DataLoader

def detection_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    return torch.stack(imgs,0), targets

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


