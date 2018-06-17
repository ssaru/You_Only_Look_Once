import os

import torch
import torch.utils.data as data
from PIL import Image

from convertYolo.Format import YOLO as cvtYOLO
from convertYolo.Format import VOC as cvtVOC

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

    CLASSES = "./voc.names"
    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        with open("./voc.names") as f:
            self.classes = f.read().splitlines()

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data =self.cvtData()

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))

    def cvtData(self):

        result = []
        voc = cvtVOC()
        yolo = cvtYOLO(os.path.abspath(self.CLASSES))
        flag, data =voc.parse(os.path.join(self.root, self.LABEL_FOLDER))

        try:

            if flag:
                flag, data =yolo.generate(data)

                keys = list(data.keys())
                keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))

                for key in keys:
                    contents = list(filter(None, data[key].split("\n")))
                    result.append({os.path.join(self.root, self.IMAGE_FOLDER, "".join([key, self.IMG_EXTENSIONS])) : contents})

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
                    x of center point,
                    y of center point,
                    width represented ratio of image width,
                    height represented ratio of image height
                ]
            ]
        """
        key = list(self.data[index].keys())[0]
        print(key)
        img = Image.open(key).convert('RGB')
        target = self.data[index][key]

        if self.transform is not None:
            img = self.transform(img)
            pass

        if self.target_transform is not None:
            # Future works
            pass

        return img, target
