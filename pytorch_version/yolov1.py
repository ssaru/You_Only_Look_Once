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

# Hyper parameters
num_epochs = 16000
num_classes = 1
batch_size = 64
learning_rate = 5e-5

dropout_prop = 0.5

# visdom function

def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 1)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(viz, iteration, loss, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 1)).cpu() * iteration,
        Y=torch.Tensor([loss]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def one_hot(output , label):

    label = label.cpu().data.numpy()
    b, s1, s2, c = output.shape
    dst = np.zeros([b,s1,s2,c], dtype=np.float32)

    for k in range(b):
        for i in range(s1):
            for j in range(s2):

                dst[k][i][j][int(label[k][i][j])] = 1.

    return torch.from_numpy(dst)

num_classes = 1
def detection_collate_for_small_yolo(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    targets = []
    imgs = []
    sizes = []
    for sample in batch:
        imgs.append(sample[0])
        #sizes.append(sample[2])
        
        np_label = np.zeros((7,7,6), dtype=np.float32)
        for i in range(7):
            for j in range(7):
                np_label[i][j][1] = (num_classes-1)
        print(len(sample[1]))
        for object in sample[1]:
            objectness=1
            cls = object[0]
            x_ratio = object[1]
            y_ratio = object[2]
            w_ratio = object[3]
            h_ratio = object[4]
            shape_np = object[5]
            
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
        print(type(np_label))
        print(type(shape_np))
        print(type(label))
        shape = torch.Tensor(shape_np)
        sizes.append(shape)
        
    return torch.stack(imgs,0), torch.stack(targets, 0), torch.stack(sizes, 0)

# Convolutional neural network (two convolutional layers)
class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        # LAYER 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), #, momentum=0.01
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192, momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 4
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer14 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer15 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 5
        self.layer17 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer18 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer19 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer20 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer21 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer22 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())

        # LAYER 6
        self.layer23 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer24 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prop)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 7*7*((5+5)+num_classes)),
            nn.Dropout(dropout_prop),
        )


        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                #nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        out = out.reshape((-1,7,7,((5+5)+num_classes)))

        return out

def detection_loss(output, target):
    # hyper parameter

    lambda_coord = 5
    lambda_noobj = 0.5

    # check batch size
    b, _, _, _ = target.shape

    # class loss
    MSE_criterion = nn.MSELoss()

    # output tensor slice
    objness1_output = output[:, :, :, 0]
    x_offset1_output = output[:, :, :, 1]
    y_offset1_output = output[:, :, :, 2]
    width_ratio1_output = output[:, :, :, 3]
    height_ratio1_output = output[:, :, :, 4]
    objness2_output = output[:, :, :, 5]
    x_offset2_output = output[:, :, :, 6]
    y_offset2_output = output[:, :, :, 7]
    width_ratio2_output = output[:, :, :, 8]
    height_ratio2_output = output[:, :, :, 9]
    class_output = output[:, :, :, 10:]

    # label tensor slice
    objness_label = target[:, :, :, 0]
    noobjness_label = torch.neg(torch.add(objness_label, -1))

    class_label = target[:, :, :, 1]

    y_one_hot = one_hot(class_output, class_label).cuda()

    #print(y_one_hot)
    #print(y_one_hot.shape)
    #exit()

    x_offset_label = target[:, :, :, 2]
    y_offset_label = target[:, :, :, 3]
    width_ratio_label = target[:, :, :, 4]
    height_ratio_label = target[:, :, :, 5]

    obj_coord1_loss = lambda_coord * \
                      torch.sum(objness_label *
                                (torch.pow(x_offset1_output - x_offset_label, 2) +
                                 torch.pow(y_offset1_output - y_offset_label, 2)))

    obj_size1_loss = lambda_coord * \
                     torch.sum(objness_label *
                               (torch.pow(width_ratio1_output - torch.sqrt(width_ratio_label), 2) +
                                torch.pow(height_ratio1_output - torch.sqrt(height_ratio_label), 2)))

    obj_coord2_loss = lambda_coord * \
                      torch.sum(objness_label *
                                (torch.pow(x_offset2_output - x_offset_label, 2) +
                                 torch.pow(y_offset2_output - y_offset_label, 2)))

    obj_size2_loss = lambda_coord * \
                     torch.sum(objness_label *
                               (torch.pow(torch.sqrt(width_ratio2_output) - torch.sqrt(width_ratio_label), 2) +
                                torch.pow(torch.sqrt(height_ratio2_output) - torch.sqrt(height_ratio_label), 2)))



    obj_class_loss = torch.sum(objness_label * MSE_criterion(class_output, y_one_hot))


    noobj_class_loss = lambda_noobj * torch.sum(noobjness_label * MSE_criterion(class_output, y_one_hot))

    objness1_loss = torch.sum(objness_label * torch.pow(objness1_output - objness_label, 2))
    objness2_loss = torch.sum(objness_label * torch.pow(objness2_output - objness_label, 2))

    total_loss = (obj_coord1_loss + obj_size1_loss + obj_coord2_loss + obj_size2_loss + noobj_class_loss +
                  obj_class_loss + objness1_loss + objness2_loss) / b

    return total_loss, obj_coord1_loss, obj_size1_loss, obj_coord2_loss, obj_size2_loss, obj_class_loss, noobj_class_loss, objness1_loss, objness2_loss

def visualize_weights_distribution(net):

    attr = net.__dict__["_modules"]
    keys = attr.keys()

    np_parmas = {}

    for key in keys:
        key_attr = eval("".join(["net", ".", key, ".", "state_dict()"]))
        for ind in key_attr:
            if "weight" in ind:
                np_parmas["".join([key, "_weights"])] = key_attr[ind].data.numpy()
            if "bias" in ind:
                np_parmas["".join([key, "_bias"])] = key_attr[ind].data.numpy()

    np_params_keys = np_parmas.keys()

    for key in np_params_keys:

        # Visualize Convolutional Layer
        if "layer" in key and "bias" not in key:

            c_out, c_in, w, h = np_parmas[key].shape

            plt.figure(figsize=(20, 20))

            cnt = 1
            # range(c_out)
            for _out in range(10):
                # range(c_in)
                for _in in range(c_in if 10 > c_in else 10):
                    plt.subplot(10, 10, cnt)
                    cnt += 1
                    n, bins, patches = plt.hist(np_parmas[key][_out, _in, :, :])


            print("Save Fig in {}".format(key))
            plt.savefig("".join([key, ".png"]))
            plt.close()
            cnt = 0

        # Visualize  Fully Connected Layer
        """
        if "fc" in key:
            c_in, c_out = np_parmas[key].shape
            plt.figure(figsize=(20, 20))

            cnt = 1
            plt.subplot(1,1,1)
            n, bins, patches = plt.hist(np_parmas[key], bins=10)
            plt.savefig("".join([key, ".png"]))
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
