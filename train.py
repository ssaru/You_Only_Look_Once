import sys
import os

import argparse
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

import yolov1

from PIL import Image
from torchsummary.torchsummary import summary
from utilities.dataloader import detection_collate
from utilities.dataloader import VOC

from utilities.utils import save_checkpoint
from utilities.utils import create_vis_plot
from utilities.utils import update_vis_plot
from utilities.utils import visualize_GT
from utilities.augmentation import Augmenter
from yolov1 import detection_loss_4_yolo
from imgaug import augmenters as iaa

warnings.filterwarnings("ignore")
#plt.ion()   # interactive mode

parser = argparse.ArgumentParser(description='YOLO v1.')
parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, voc', default='voc')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--class_path',                type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=448)
parser.add_argument('--input_width',               type=int,   help='input width', default=448)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=15)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=16000)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--dropout',                   type=float, help='dropout probability', default=0.5)
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='./')

# flag
parser.add_argument('--use_augmentation',          type=bool,  help='Image Augmentation', default=True)
parser.add_argument('--use_visdom',                type=bool,  help='visdom board', default=True)
parser.add_argument('--use_summary',               type=bool,  help='descripte Model summary', default=True)
parser.add_argument('--use_gtcheck',               type=bool,  help='Ground Truth check flag', default=False)

# develop
parser.add_argument('--num_class',                 type=int,  help='number of class', default=5, required=True)
args = parser.parse_args()

# model = torch.nn.DataParallel(net, device_ids=[0]).cuda()
def train(params):

    # future work variable
    dataset             = params["dataset"]
    input_height        = params["input_height"]
    input_width         = params["input_width"]

    data_path           = params["data_path"]
    class_path          = params["class_path"]
    batch_size          = params["batch_size"]
    num_epochs          = params["num_epochs"]
    learning_rate       = params["lr"]
    dropout             = params["dropout"]
    num_gpus            = [ i for i in range(params["num_gpus"])]
    checkpoint_path     = params["checkpoint_path"]

    USE_VISDOM          = params["use_visdom"]
    USE_SUMMARY         = params["use_summary"]
    USE_AUGMENTATION    = params["use_augmentation"]
    USE_GTCHECKER       = params["use_gtcheck"]

    num_class           = params["num_class"]

    with open(class_path) as f:
        class_list = f.read().splitlines()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if USE_VISDOM:
        viz = visdom.Visdom(use_incoming_socket=False)
        vis_title = 'Yolo V1 Deepbaksu_vision (feat. martin, visionNoob) PyTorch on ' + 'VOC'
        vis_legend = ['Train Loss']
        iter_plot = create_vis_plot(viz, 'Iteration', 'Total Loss', vis_title, vis_legend)
        coord1_plot = create_vis_plot(viz, 'Iteration', 'coord1', vis_title, vis_legend)
        size1_plot = create_vis_plot(viz, 'Iteration', 'size1', vis_title, vis_legend)
        noobjectness1_plot = create_vis_plot(viz, 'Iteration', 'noobjectness1', vis_title, vis_legend)
        objectness1_plot = create_vis_plot(viz, 'Iteration', 'objectness1', vis_title, vis_legend)
        obj_cls_plot = create_vis_plot(viz, 'Iteration', 'obj_cls', vis_title, vis_legend)

    # 2. Data augmentation setting
    if (USE_AUGMENTATION):
        seq = iaa.SomeOf(2, [
            iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
            iaa.Affine(
                translate_px={"x": 3, "y": 10},
                scale=(0.9, 0.9)
            ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
            iaa.AdditiveGaussianNoise(scale=0.1 * 255),
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
            iaa.Affine(rotate=45),
            iaa.Sharpen(alpha=0.5)
        ])
    else:
        seq = iaa.Sequential([])

    composed = transforms.Compose([Augmenter(seq)])

    # 3. Load Dataset
    # composed
    # transforms.ToTensor
    train_dataset = VOC(root=data_path, transform=composed, class_path=class_path)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=detection_collate)

    # 5. Load YOLOv1
    net = yolov1.YOLOv1(params={"dropout" : dropout, "num_class" : num_class})
    model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()

    if USE_SUMMARY:
        summary(model, (3, 448, 448))

    # 7.Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):

        if (epoch == 200) or (epoch == 400) or (epoch == 600) or (epoch == 20000) or (epoch == 30000):
            scheduler.step()

        for i, (images, labels, sizes) in enumerate(train_loader):

            if USE_GTCHECKER:
                visualize_GT(images, labels, class_list)
                exit()

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
            objness1_loss = detection_loss_4_yolo(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(
                    'Epoch ,[{}/{}] ,Step ,[{}/{}] ,lr ,{} ,total_loss ,{:.4f} ,coord1 ,{} ,size1 ,{} ,noobj_clss ,{} ,objness1 ,{} ,'
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

                if USE_VISDOM:
                    update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), loss.item(), iter_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_coord1_loss, coord1_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_size1_loss, size1_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), obj_class_loss, obj_cls_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), noobjness1_loss, noobjectness1_plot, None,
                                    'append')
                    update_vis_plot(viz, (epoch + 1) * batch_size + (i + 1), objness1_loss, objectness1_plot, None,
                                    'append')

        if (epoch % 300) == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "YOLOv1",
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False, filename=os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(epoch)))

def main():
    params = {
        "mode"              : args.mode,
        "dataset"           : args.dataset,
        "data_path"         : args.data_path,
        "class_path"        : args.class_path,
        "input_height"      : args.input_height,
        "input_width"       : args.input_width,
        "batch_size"        : args.batch_size,
        "num_epochs"        : args.num_epochs,
        "lr"                : args.learning_rate,
        "dropout"           : args.dropout,
        "num_gpus"          : args.num_gpus,
        "checkpoint_path"   : args.checkpoint_path,

        "use_visdom"        : args.use_visdom,
        "use_summary"       : args.use_summary,
        "use_augmentation"  : args.use_augmentation,

        "num_class"         : args.num_class,
        "use_gtcheck"       : args.use_gtcheck

    }

    if params["mode"] == "train":
        train(params)
    elif params["mode"] == "test":
        # Future Work
        pass

if __name__ == '__main__':
    main()