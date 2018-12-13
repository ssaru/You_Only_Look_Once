# -*- coding:utf-8 -*-

import numpy as np
from utilities.utils import CvtCoordsXXYY2XYWH
from utilities.utils import CvtCoordsXYWH2XXYY
from utilities.utils import GetImgaugStyleBBoxes
from utilities.utils import GetYoloStyleBBoxes
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def augmentImage(image, normed_lxywhs, image_width, image_height, seq):

    bbs = GetImgaugStyleBBoxes(normed_lxywhs, image_width, image_height)

    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]

    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()

    if(False):
        image_before = bbs.draw_on_image(image, thickness=5)
        image_after = bbs_aug.draw_on_image(image_aug, thickness=5, color=[0, 0, 255])

        fig = plt.figure(1, (10., 10.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    normed_bbs_aug = GetYoloStyleBBoxes(normed_lxywhs, bbs_aug, image_width, image_height)

    return image_aug, normed_bbs_aug


class Augmenter(object):

    def __init__(self, seq):
        self.seq = seq

    def __call__(self, sample):

        image = sample[0]  # PIL image
        normed_lxywhs = sample[1]
        image_width, image_height = image.size

        image = np.array(image)  # PIL image to numpy array

        image_aug, normed_bbs_aug = augmentImage(image, normed_lxywhs, image_width, image_height, self.seq)

        image_aug = Image.fromarray(image_aug)  # numpy array to PIL image Again!
        return image_aug, normed_bbs_aug
