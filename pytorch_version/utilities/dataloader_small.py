import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.utils.data as data
from PIL import Image

from convertYolo.Format import YOLO as cvtYOLO
from convertYolo.Format import VOC as cvtVOC
"""
try:
    from convertYolo.Format import YOLO as cvtYOLO
    from convertYolo.Format import VOC as cvtVOC
except Exception: # ImportError
    import convertYolo.Format as Format
    cvtYOLO = Format.YOLO
    cvtVOC = Format.VOC
"""

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

    #CLASSES = "./voc.names"
    CLASSES = "./person.names"
    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'

    def __init__(self, root, train=True, transform=None, target_transform=None, resize=448, cls_option=False, selective_cls=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize
        self.cls_option = cls_option
        self.selective_cls = selective_cls

        with open("./voc.names") as f:
            self.classes = f.read().splitlines()

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data = self.cvtData()

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))

    def cvtData(self):

        result = []
        voc = cvtVOC()
        yolo = cvtYOLO(os.path.abspath(self.CLASSES))
        flag, data =voc.parse(os.path.join(self.root, self.LABEL_FOLDER), cls_option=self.cls_option, selective_cls=self.selective_cls)
        #print(flag, data)
        #exit()

        try:

            if flag:
                flag, data =yolo.generate(data)

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

                    result.append({os.path.join(self.root, self.IMAGE_FOLDER, "".join([key, self.IMG_EXTENSIONS])) : target})

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
        #print("INDEX : {}".format(index))
        key = list(self.data[index].keys())[0]
        #print("KEY : {}".format(key))
        img = Image.open(key).convert('RGB')
        
        current_shape = img.size
        #print('current_shape:',current_shape)
        
        img = img.resize((self.resize_factor, self.resize_factor))
        target = self.data[index][key]

        if self.transform is not None:
            img = self.transform(img)
            pass

        if self.target_transform is not None:
            # Future works
            pass
        
        
        return img, target, current_shape
