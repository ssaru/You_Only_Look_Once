import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision
import torch.utils.data as data
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import numpy as np

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
    #CLASSES = "./person.names"
    CLASSES = "./5class.names"
    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'

    def __init__(self, root, train=True, transform=None, target_transform=None, resize=448, cls_option=False, selective_cls=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize
        self.cls_option = cls_option
        self.selective_cls = selective_cls

        with open("./5class.names") as f:
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
        yolo = cvtYOLO(os.path.abspath(self.CLASSES))
        flag, self.dict_data =voc.parse(os.path.join(self.root, self.LABEL_FOLDER), cls_option=self.cls_option, selective_cls=self.selective_cls)
        #print(flag, data)
        #exit()

        try:

            if flag:
                flag, data =yolo.generate(self.dict_data)

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


        key = list(self.data[index].keys())[0]

        img = Image.open(key).convert('RGB')
        
        current_shape = img.size
        
        img = img.resize((self.resize_factor, self.resize_factor))
        img = np.array(img.getdata(), dtype=np.float).reshape(img.size[0], img.size[1], 3)

        target = self.data[index][key]

        # for debug & visualization
        """
        _image = img.astype(dtype=np.uint8)
        _image = Image.fromarray(_image, "RGB")
        draw = ImageDraw.Draw(_image)


        for obj in target:

            cls = obj[0]
            x_center = obj[1]
            y_center = obj[2]
            w_ratio = obj[3]
            h_ratio = obj[4]

            width = int(w_ratio * self.resize_factor)
            height = int(h_ratio * self.resize_factor)

            xmin = int(x_center * self.resize_factor - width/2)
            ymin = int(y_center * self.resize_factor - height/2)
            xmax = xmin + width
            ymax = ymin + height

            cls = self.classes[int(cls)]

            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue")
            draw.text((xmin, ymin), cls)


        plt.imshow(_image)
        plt.show()
        """


        if self.transform is not None:
            #img, target = self.transform([img, target])
            #img = torchvision.transforms.ToTensor()(img)
            img = img.astype(dtype=np.uint8)
            print(img.shape)
            print(img.dtype)
            plt.imshow(img)
            plt.show()
            img = img.astype(dtype=np.float64)
            img = self.transform(img)
            print(img.shape)
            image = img.type(torch.ByteTensor).view(448,448,3)
            image = image.cpu().numpy()
            plt.imshow(image)
            plt.show()
            exit()


        else:
            img = torch.FloatTensor(img)
            img = torch.div(img, 255)

        if self.target_transform is not None:
            # Future works
            pass
        
        
        return img, target