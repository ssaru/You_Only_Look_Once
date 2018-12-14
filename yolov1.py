# -*- coding:utf-8 -*-

import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchsummary.torchsummary import summary
from utilities import dataloader
from dataloader import VOC

import numpy as np
import matplotlib.pyplot as plt

import visdom


class YOLOv1(nn.Module):
    def __init__(self, params):

        self.dropout_prop = params["dropout"]
        self.num_classes = params["num_class"]

        super(YOLOv1, self).__init__()
        # LAYER 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
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
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prop)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 7 * 7 * ((5) + self.num_classes))
        )

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
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
        out = out.reshape((-1, 7, 7, ((5) + self.num_classes)))
        out[:, :, :, 0] = torch.sigmoid(out[:, :, :, 0])  # sigmoid to objness1_output
        out[:, :, :, 5:] = torch.sigmoid(out[:, :, :, 5:])  # sigmoid to class_output

        return out


# def detection_loss_4_yolo(output, target):
def detection_loss_4_yolo(output, target, device):
    from utilities.utils import one_hot

    # hyper parameter

    lambda_coord = 5
    lambda_noobj = 0.5

    # check batch size
    b, _, _, _ = target.shape
    _, _, _, n = output.shape

    # output tensor slice
    # output tensor shape is [batch, 7, 7, 5 + classes]
    objness1_output = output[:, :, :, 0]
    x_offset1_output = output[:, :, :, 1]
    y_offset1_output = output[:, :, :, 2]
    width_ratio1_output = output[:, :, :, 3]
    height_ratio1_output = output[:, :, :, 4]
    class_output = output[:, :, :, 5:]

    num_cls = class_output.shape[-1]

    # label tensor slice
    objness_label = target[:, :, :, 0]
    x_offset_label = target[:, :, :, 1]
    y_offset_label = target[:, :, :, 2]
    width_ratio_label = target[:, :, :, 3]
    height_ratio_label = target[:, :, :, 4]
    class_label = one_hot(class_output, target[:, :, :, 5], device)

    noobjness_label = torch.neg(torch.add(objness_label, -1))

    obj_coord1_loss = lambda_coord * \
                      torch.sum(objness_label *
                        (torch.pow(x_offset1_output - x_offset_label, 2) +
                                    torch.pow(y_offset1_output - y_offset_label, 2)))

    obj_size1_loss = lambda_coord * \
                     torch.sum(objness_label *
                               (torch.pow(width_ratio1_output - torch.sqrt(width_ratio_label), 2) +
                                torch.pow(height_ratio1_output - torch.sqrt(height_ratio_label), 2)))

    objectness_cls_map = objness_label.unsqueeze(-1)

    for i in range(num_cls - 1):
        objectness_cls_map = torch.cat((objectness_cls_map, objness_label.unsqueeze(-1)), 3)

    obj_class_loss = torch.sum(objectness_cls_map * torch.pow(class_output - class_label, 2))

    noobjness1_loss = lambda_noobj * torch.sum(noobjness_label * torch.pow(objness1_output - objness_label, 2))
    objness1_loss = torch.sum(objness_label * torch.pow(objness1_output - objness_label, 2))

    total_loss = (obj_coord1_loss + obj_size1_loss + noobjness1_loss + objness1_loss + obj_class_loss)
    total_loss = total_loss / b

    return total_loss, obj_coord1_loss / b, obj_size1_loss / b, obj_class_loss / b, noobjness1_loss / b, objness1_loss / b
