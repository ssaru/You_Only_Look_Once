# -*- coding:utf-8 -*-

import sys
import os

import torch
import torch.utils.data as data
import numpy as np

from PIL import Image

from convertYolo.Format import YOLO as cvtYOLO
from convertYolo.Format import VOC as cvtVOC

import torchvision
import torchvision.transforms as transforms

# develop
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))


def detection_collate(batch):
    """ `Puts each data field into a tensor with outer dimension batch size`

    Args:
        batch : batch data ``batch[0]`` : image, ``batch[1]`` : label, ``batch[3]`` : size

    Return:
        image tensor, label tensor

    Future work:
        return value(torch.stack) change to Torch.FloatTensor()
    """

    targets = []
    imgs = []
    sizes = []

    for sample in batch:
        imgs.append(sample[0])

        # for drawing box
        # if using batch it should keep original image size.
        sizes.append(sample[2])

        np_label = np.zeros((7, 7, 6), dtype=np.float32)
        for object in sample[1]:
            objectness = 1
            classes = object[0]
            x_ratio = object[1]
            y_ratio = object[2]
            w_ratio = object[3]
            h_ratio = object[4]

            # can be acuqire grid (x,y) index when divide (1/S) of x_ratio
            scale_factor = (1 / 7)
            grid_x_index = int(x_ratio // scale_factor)
            grid_y_index = int(y_ratio // scale_factor)
            x_offset = (x_ratio / scale_factor) - grid_x_index
            y_offset = (y_ratio / scale_factor) - grid_y_index

            # insert object row in specific label tensor index as (x,y)
            # object row follow as
            # [objectness, class, x offset, y offset, width ratio, height ratio]
            np_label[grid_x_index][grid_y_index] = np.array([objectness, x_offset, y_offset, w_ratio, h_ratio, classes])

        label = torch.from_numpy(np_label)
        targets.append(label)

    return torch.stack(imgs, 0), torch.stack(targets, 0), sizes


class VOC(data.Dataset):
    """ `VOC PASCAL Object Detection Challenge <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>_ Dataset `

    Args:
        root (string): Root directory of dataset where ``VOCdevkit/VOC20XX/``
        train (bool, optional): If True, creates dataset from ``training``folder,
            otherwise from ``test``folder
        transform (callable, optional): A function/trainsform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        => future works. now it's not implementation
           It's only support transforms.ToTensor()
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        => future works. now it's not implementation

    """

    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'

    def __init__(self, root, train=True, transform=None, target_transform=None, resize=448, class_path='./voc.names'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize
        self.class_path = class_path

        with open(class_path) as f:
            self.classes = f.read().splitlines()

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data = self.cvtData()

    def _check_exists(self):
        print("Image Folder : {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
        print("Label Folder : {}".format(os.path.join(self.root, self.LABEL_FOLDER)))

        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))

    def cvtData(self):

        result = []
        voc = cvtVOC()

        yolo = cvtYOLO(os.path.abspath(self.class_path))
        flag, self.dict_data = voc.parse(os.path.join(self.root, self.LABEL_FOLDER))

        try:

            if flag:
                flag, data = yolo.generate(self.dict_data)

                keys = list(data.keys())
                keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))

                for key in keys:
                    contents = list(filter(None, data[key].split("\n")))
                    target = []
                    for i in range(len(contents)):
                        tmp = contents[i]
                        tmp = tmp.split(" ")
                        for j in range(len(tmp)):
                            tmp[j] = float(tmp[j])
                        target.append(tmp)

                    result.append({os.path.join(self.root, self.IMAGE_FOLDER, "".join([key, self.IMG_EXTENSIONS])): target})

                return result

        except Exception as e:
            raise RuntimeError("Error : {}".format(e))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int) : Index

        Returns:
            tuple: Tuple(image, target). target is the object returned by YOLO annotation as
            [
                [   class,
                  print("Hello")  x of center point,
                    y of center point,
                    width represented ratio of image width,
                    height represented ratio of image height
                ]
            ]

        """
        key = list(self.data[index].keys())[0]

        img = Image.open(key).convert('RGB')
        current_shape = img.size
        img = img.resize((self.resize_factor, self.resize_factor))

        target = self.data[index][key]

        if self.transform is not None:
            img, aug_target = self.transform([img, target])
            img = torchvision.transforms.ToTensor()(img)

        if self.target_transform is not None:
            # Future works
            pass

        return img, aug_target, current_shape
