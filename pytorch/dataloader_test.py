import torch
import torchvision.transforms as transforms

from pytorch.dataset import VOC

batch_size = 1
root ="/media/martin/Martin/DataSet/VOC/VOCdevkit/VOC2008/"
train_dataset = VOC(root=root, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

total_step = len(train_loader)

for i, (images, lables) in enumerate(train_loader):
    print(lables)
    print(images)
    break