import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchsummary.torchsummary import summary
from dataloader import VOC

import numpy as np

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
num_classes = 20
batch_size = 1
learning_rate = 0.001

def calc_loss(prediction, y_hat):
    lambda_obj = 5
    lambda_noobj = 0.5

    MSE_criterion = nn.MSELoss()
    CLASS_criterion = nn.CrossEntropyLoss()

    coordinate_loss1 = torch.pow( torch.sub(get_axis_array(y_hat, 2), get_axis_array(prediction, 1)), 2) + \
                       torch.pow( torch.sub(get_axis_array(y_hat, 3), get_axis_array(prediction, 2)), 2)
    coordinate_loss2 = torch.pow( torch.sub(get_axis_array(y_hat, 2), get_axis_array(prediction, 5)), 2) + \
                       torch.pow( torch.sub(get_axis_array(y_hat, 3), get_axis_array(prediction, 6)), 2)



    pass



def get_axis_array(x, axis):

    width, height, channels = x.shape

    out = np.zeros((width, height))

    for i in range(width):
        for j in range(height):
            out[i][j] = x[i][j][axis]

    return torch.from_numpy(out)

def detection_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])

        np_label = np.zeros((7,7,6), dtype=np.float32)
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

# VOC Pascal Dataset
train_dataset = VOC(root = "/media/martin/keti_martin/Martin/DataSet/VOC_Pascal/VOCdevkit/VOC2008/",
                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           collate_fn=detection_collate)

# Convolutional neural network (two convolutional layers)
class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        # LAYER 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=1),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 4
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.layer14 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer15 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 5
        self.layer17 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.layer18 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer19 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.layer20 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer21 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer22 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        # LAYER 6
        self.layer23 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer24 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 7*7*((5+5)+num_classes)),
            nn.Dropout(),
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out = self.layer19(out)
        out = self.layer20(out)
        out = self.layer21(out)
        out = self.layer22(out)
        out = self.layer23(out)
        out = self.layer24(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.reshape((-1,7,7,30), 1)

        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv1().to(device)

summary(model, (3, 448,448))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        print(outputs.shape)
        print(images.shape)
        print(labels.shape)
        exit()

        """
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                  
        
        """

"""
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
"""