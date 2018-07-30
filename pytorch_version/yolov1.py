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
num_epochs = 10000
num_classes = 21
batch_size = 32
learning_rate = 1e-5

def one_hot(output , label):

    label = label.cpu().data.numpy()
    b, s1, s2, c = output.shape
    dst = np.zeros([b,s1,s2,c], dtype=np.float32)

    for k in range(b):
        for i in range(s1):
            for j in range(s2):
                dst[k][i][j][int(label[k][i][j])] = 1.

    return torch.from_numpy(dst)


def detection_loss(output, target):

    # hyper parameter
    lambda_coord = 5
    lambda_noobj = 0.5

    # check batch size
    b, _, _, _ = target.shape

    # class loss
    MSE_criterion = nn.MSELoss()

    # output tensor slice
    objness1_output = output[:,:,:,0]
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

    x_offset_label = target[:, :, :, 2]
    y_offset_label = target[:, :, :, 3]
    width_ratio_label = target[:, :, :, 4]
    height_ratio_label = target[:, :, :, 5]

    obj_coord1_loss = lambda_coord * \
                  torch.sum( objness_label *
                             (torch.pow(x_offset1_output - x_offset_label, 2) +
                              torch.pow(y_offset1_output - y_offset_label,2)))

    obj_size1_loss = lambda_coord * \
                 torch.sum(objness_label *
                           (torch.pow(torch.sqrt(torch.sqrt(torch.pow(width_ratio1_output, 2))) -
                                      torch.sqrt(torch.sqrt(torch.pow(width_ratio_label, 2))), 2) +
                           torch.pow(torch.sqrt(torch.sqrt(torch.pow(height_ratio1_output, 2))) -
                                     torch.sqrt(torch.sqrt(torch.pow(height_ratio_label,2))), 2)))

    obj_coord2_loss = lambda_coord * \
                      torch.sum(objness_label *
                                (torch.pow(x_offset2_output - x_offset_label, 2) +
                                 torch.pow(y_offset2_output - y_offset_label, 2)))

    obj_size2_loss = lambda_coord * \
                     torch.sum(objness_label *
                               (torch.pow(torch.sqrt(torch.sqrt(torch.pow(width_ratio2_output,2))) -
                                          torch.sqrt(torch.sqrt(torch.pow(width_ratio_label, 2))), 2) +
                                torch.pow(torch.sqrt(torch.sqrt(torch.pow(height_ratio2_output, 2))) -
                                          torch.sqrt(torch.sqrt(torch.pow(height_ratio_label, 2))),2)))

    noobj_class_loss = lambda_noobj * torch.sum(noobjness_label * MSE_criterion(class_output, y_one_hot))

    objness1_loss = torch.sum(torch.pow(objness1_output - objness_label, 2))
    objness2_loss = torch.sum(torch.pow(objness2_output - objness_label, 2))

    total_loss = (obj_coord1_loss + obj_size1_loss + obj_coord2_loss + obj_size2_loss + noobj_class_loss + \
                 objness1_loss + objness2_loss) / b

    """
    print("obj_coord1_loss : {}".format(obj_coord1_loss))
    print("obj_size1_loss : {}".format(obj_size1_loss))
    print("obj_coord2_loss : {}".format(obj_coord2_loss))
    print("obj_size2_loss : {}".format(obj_size2_loss))
    print("noobj_class_loss : {}".format(noobj_class_loss))
    print("objness1_loss : {}".format(objness1_loss))
    print("objness2_loss : {}".format(objness2_loss))
    """

    return total_loss


def calc_loss(prediction, y_hat):
    lambda_obj = 5
    lambda_noobj = 0.5

    MSE_criterion = nn.MSELoss()
    CLASS_criterion = nn.CrossEntropyLoss()

    COORDINATE_LOSS1 = []
    COORDINATE_LOSS2 = []

    SIZE_LOSS1 = []
    SIZE_LOSS2 = []

    OBJNESS_CLS_LOSS = []
    NOOBJNESS_CLS_LOSS = []
    OBJNESS_LOSS1 = []
    OBJNESS_LOSS2 = []

    objness_check = np.array([0], dtype=np.float32)
    cpu_label = y_hat.cpu().data.numpy()

    objness_true = torch.cuda.FloatTensor([1])
    objness_false = torch.cuda.FloatTensor([0])

    noobjness_true = torch.cuda.FloatTensor([1])
    noobjness_false = torch.cuda.FloatTensor([0])

    background_cls_true = torch.cuda.FloatTensor([0])
    background_cls_false = torch.cuda.FloatTensor([1])

    coordinate_loss1 = torch.cuda.FloatTensor([0])
    coordinate_loss2 = torch.cuda.FloatTensor([0])
    size_loss1 = torch.cuda.FloatTensor([0])
    size_loss2 = torch.cuda.FloatTensor([0])
    obj_cls_loss = torch.cuda.FloatTensor([0])
    noobj_cls_loss = torch.cuda.FloatTensor([0])
    objness_loss1 = torch.cuda.FloatTensor([0])
    objness_loss2 = torch.cuda.FloatTensor([0])

    for batch in range(len(y_hat)):
        for i in range(len(y_hat[0])):
            for j in range(len(y_hat[0][0])):
                #print(get_axis_array(y_hat[batch], i, j))

                label_ = get_axis_array(y_hat[batch], i, j)
                clabel = get_axis_array(cpu_label[batch], i, j)
                pred_ = get_axis_array(prediction[batch], i, j)

                """
                print("label_ : {}".format(label_))
                print("pred_ : {}".format(pred_))


                print("HERE")
                print(label_)
                print(label_.dtype)
                print(label_.data[0])
                """

                check = clabel[0]

                if check == objness_check:
                    objness = objness_true
                    noobjness = noobjness_false
                    objness_flag = True
                else:
                    objness = objness_false
                    noobjness = noobjness_true
                    objness_flag = False

                # label elements
                # [objectness, class, x offset, y offset, width ratio, height ratio]

                """
                print("objness type : {}".format(objness.dtype))
                print("pred_ type : {}".format(pred_[1].dtype))
                print("label__ type : {}".format(label_[2].dtype))
                """

                # objness case
                point_loss_1 = objness * (torch.pow(pred_[1] - label_[2], 2) + torch.pow(pred_[2] - label_[3], 2))
                size_loss_1 = objness * (torch.pow(torch.sqrt(torch.sqrt(torch.pow(pred_[3], 2))) -
                                                   torch.sqrt(label_[4]), 2) +
                                         torch.pow(torch.sqrt(torch.sqrt(torch.pow(pred_[4], 2))) - torch.sqrt(label_[5]),2))

                point_loss_2 = objness * (torch.pow(pred_[6] - label_[2], 2) + torch.pow(pred_[7] - label_[3], 2))

                size_loss_2 = objness * (torch.pow(torch.sqrt(torch.sqrt(torch.pow(pred_[8],2))) -
                                                   torch.sqrt(label_[4]), 2) +
                                         torch.pow(torch.sqrt(torch.sqrt(torch.pow(pred_[9],2))) - torch.sqrt(label_[5]), 2))

                """
                elem = {"width pred" : pred_[8],
                        "width label": label_[4],
                        "height pred": pred_[9],
                        "height label": label_[5]}
                print("objness :{}".format(objness))
                print("width total :{}".format(torch.pow(torch.sqrt(pred_[8]) - torch.sqrt(label_[4]), 2)))
                print("width pred :{}".format(pred_[8]))
                print("width pred sqrt :{}".format(torch.sqrt(pred_[8])))
                print("width label :{}".format(label_[4]))
                print("width label sqrt :{}".format(torch.sqrt(label_[4])))
                print("height total :{}".format(torch.pow(torch.sqrt(pred_[9]) - torch.sqrt(label_[5]), 2)))
                print("height pred :{}".format(pred_[9]))
                print("height pred sqrt :{}".format(torch.sqrt(pred_[9])))
                print("height label :{}".format(label_[5]))
                print("height label sqrt :{}".format(torch.sqrt(label_[5]), 2))
                print("size_loss_2 : {}, \nelements : {}\n pred : {} \nlabel : {}".format(size_loss_2, elem, pred_,label_))
                print("\n\n\n\n")
                """


                #cls_pred = pred_[10:].view([1, 20])
                cls_pred = pred_[10:]

                cls_pred = cls_pred.view([1, 21]).float()
                cls_label = label_[1].view([1]).long()

                """
                print("prediction : {}".format(cls_pred))
                print("Shape of Prediction : {}".format(cls_pred.shape))
                print("label : {}".format(cls_label))
                print("Shape of label : {}".format(cls_label.shape))


                print("objness type : {}".format(objness.dtype))
                print("pred_ type : {}".format(cls_pred.dtype))
                print("label__ type : {}".format(cls_label.dtype))
                """

                objness_cls_loss = objness * CLASS_criterion(cls_pred, cls_label)

                # noobjness case
                noobjness_cls_loss = noobjness * CLASS_criterion(cls_pred, cls_label)


                # objness loss case
                objness_loss_1 = objness * torch.pow(pred_[0] - label_[0], 2)
                objness_loss_2 = objness * torch.pow(pred_[5] - label_[0], 2)

                # appendix
                COORDINATE_LOSS1.append(point_loss_1)
                COORDINATE_LOSS2.append(point_loss_2)

                SIZE_LOSS1.append(size_loss_1)
                SIZE_LOSS2.append(size_loss_2)

                OBJNESS_CLS_LOSS.append(objness_cls_loss)
                NOOBJNESS_CLS_LOSS.append(noobjness_cls_loss)

                OBJNESS_LOSS1.append(objness_loss_1)
                OBJNESS_LOSS2.append(objness_loss_2)


    for loss in  COORDINATE_LOSS1:
        each_loss = torch.sum(loss)
        coordinate_loss1 += each_loss


    for loss in COORDINATE_LOSS2:
        each_loss = torch.sum(loss)
        coordinate_loss2 += each_loss


    for loss in SIZE_LOSS1:
        each_loss = torch.sum(loss)
        size_loss1 += each_loss


    for loss in SIZE_LOSS2:
        each_loss = torch.sum(loss)
        """
        print("EACH SIZE_LOSS2 : {}".format(loss))
        print("EACH SUM SIZE_LOSS2 : {}".format(each_loss))
        print()
        """

        size_loss2 = each_loss


    for loss in OBJNESS_CLS_LOSS:
        each_loss = torch.sum(loss)
        obj_cls_loss += each_loss


    for loss in NOOBJNESS_CLS_LOSS:
        each_loss = torch.sum(loss)
        noobj_cls_loss += each_loss


    for loss in OBJNESS_LOSS1:
        each_loss = torch.sum(loss)
        objness_loss1 += each_loss


    for loss in OBJNESS_LOSS2:
        each_loss = torch.sum(loss)
        objness_loss2 += each_loss


    total_loss = ((lambda_obj * coordinate_loss1) + \
                 (lambda_obj * size_loss1) + \
                 (lambda_obj * coordinate_loss2) + \
                 (lambda_obj * size_loss2) + \
                 obj_cls_loss + \
                 (lambda_noobj * noobj_cls_loss) + \
                 objness_loss1 + \
                 objness_loss2) / batch_size


    print()
    print("lambda_obj Loss :{}".format(lambda_obj))
    print("lambda_noobj Loss :{}".format(lambda_noobj))
    print("coordinate_loss1 Loss :{}".format(coordinate_loss1))
    print("size_loss1 Loss :{}".format(size_loss1))
    print("coordinate_loss2 Loss :{}".format(coordinate_loss2))
    print("size_loss2 Loss :{}".format(size_loss2))
    print("obj_cls_loss Loss :{}".format(obj_cls_loss))
    print("noobj_cls_loss Loss :{}".format(noobj_cls_loss))
    print("objness_loss1 Loss :{}".format(objness_loss1))
    print("objness_loss2 Loss :{}".format(objness_loss2))
    print("Total Loss :{}".format(total_loss))


    return total_loss



def get_axis_array(x, i, j):

    out = x[i][j][:]

    return out

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

# VOC Pascal Dataset
train_dataset = VOC(root = "/media/keti-1080ti/ketiCar/DataSet/VOC/VOCdevkit/VOC2012/",
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
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 4
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        self.layer14 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer15 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 5
        self.layer17 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        self.layer18 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer19 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        self.layer20 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer21 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer22 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU())

        # LAYER 6
        self.layer23 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        self.layer24 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 7*7*((5+5)+num_classes)),
            nn.Dropout(),
        )


    def forward(self, x):
        self.out = self.layer1(x)
        self.out = self.layer2(self.out)
        self.out = self.layer3(self.out)
        self.out = self.layer4(self.out)
        self.out = self.layer5(self.out)
        self.out = self.layer6(self.out)
        self.out = self.layer7(self.out)
        self.out = self.layer8(self.out)
        self.out = self.layer9(self.out)
        self.out = self.layer10(self.out)
        self.out = self.layer11(self.out)
        self.out = self.layer12(self.out)
        self.out = self.layer13(self.out)
        self.out = self.layer14(self.out)
        self.out = self.layer15(self.out)
        self.out = self.layer16(self.out)
        self.out = self.layer17(self.out)
        self.out = self.layer18(self.out)
        self.out = self.layer19(self.out)
        self.out = self.layer20(self.out)
        self.out = self.layer21(self.out)
        self.out = self.layer22(self.out)
        self.out = self.layer23(self.out)
        self.out = self.layer24(self.out)
        self.out = self.out.reshape(self.out.size(0), -1)
        self.out = self.fc1(self.out)
        self.out = self.fc2(self.out)
        self.out = self.out.reshape((-1,7,7,31))

        return self.out

    @property
    def loss(self):
        return self.total_loss

    def detection_loss(self, output, target):

        #output = outputs.cpu()
        #target = targets.cpu()

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

        x_offset_label = target[:, :, :, 2]
        y_offset_label = target[:, :, :, 3]
        width_ratio_label = target[:, :, :, 4]
        height_ratio_label = target[:, :, :, 5]

        self.obj_coord1_loss = lambda_coord * \
                          torch.sum(objness_label *
                                    (torch.pow(x_offset1_output - x_offset_label, 2) +
                                     torch.pow(y_offset1_output - y_offset_label, 2)))

        self.obj_size1_loss = lambda_coord * \
                         torch.sum(objness_label *
                                   (torch.pow(torch.sqrt(torch.sqrt(torch.pow(width_ratio1_output, 2))) -
                                              torch.sqrt(torch.sqrt(torch.pow(width_ratio_label, 2))), 2) +
                                    torch.pow(torch.sqrt(torch.sqrt(torch.pow(height_ratio1_output, 2))) -
                                              torch.sqrt(torch.sqrt(torch.pow(height_ratio_label, 2))), 2)))

        self.obj_coord2_loss = lambda_coord * \
                          torch.sum(objness_label *
                                    (torch.pow(x_offset2_output - x_offset_label, 2) +
                                     torch.pow(y_offset2_output - y_offset_label, 2)))

        self.obj_size2_loss = lambda_coord * \
                         torch.sum(objness_label *
                                   (torch.pow(torch.sqrt(torch.sqrt(torch.pow(width_ratio2_output, 2))) -
                                              torch.sqrt(torch.sqrt(torch.pow(width_ratio_label, 2))), 2) +
                                    torch.pow(torch.sqrt(torch.sqrt(torch.pow(height_ratio2_output, 2))) -
                                              torch.sqrt(torch.sqrt(torch.pow(height_ratio_label, 2))), 2)))

        self.noobj_class_loss = lambda_noobj * torch.sum(noobjness_label * MSE_criterion(class_output, y_one_hot))

        self.objness1_loss = torch.sum(torch.pow(objness1_output - objness_label, 2))
        self.objness2_loss = torch.sum(torch.pow(objness2_output - objness_label, 2))

        self.total_loss = (self.obj_coord1_loss + self.obj_size1_loss + self.obj_coord2_loss + self.obj_size2_loss
                           + self.noobj_class_loss + self.objness1_loss + self.objness2_loss) / b

        """
        print("obj_coord1_loss : {}".format(obj_coord1_loss))
        print("obj_size1_loss : {}".format(obj_size1_loss))
        print("obj_coord2_loss : {}".format(obj_coord2_loss))
        print("obj_size2_loss : {}".format(obj_size2_loss))
        print("noobj_class_loss : {}".format(noobj_class_loss))
        print("objness1_loss : {}".format(objness1_loss))
        print("objness2_loss : {}".format(objness2_loss))
        """
        print("LOSS : {}".format(self.total_loss))
        return self.total_loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = YOLOv1()
model = net.to(device)
net.train()

summary(model, (3, 448,448))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        print(images)

        # Forward pass
        outputs = model(images)

        """
        print(outputs)
        print(outputs.shape)
        print(labels)
        print(labels.shape)
        print("label length : {}, {}, {}, {}".format(len(labels) ,len(labels[0]), len(labels[0][0]), len(labels[0][0][0])))
        """

        #loss = calc_loss(outputs, labels)
        net.detection_loss(net.out, labels)
        loss = net.loss


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                  
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