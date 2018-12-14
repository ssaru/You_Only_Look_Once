# A PyTorch implementation of a YOLO v1 Object Detector
 Implementation of YOLO v1 object detector in PyTorch. Full tutorial can be found [here](https://deepbaksuvision.github.io/Modu_ObjectDetection/) in korean.

 Tested under Python 3.6, PyTorch 0.4.1 on Ubuntu 16.04, Windows10.

## prerequisites

- python >= 3.6
- pytorch >= 1.0.0 (1.0.0 also fine)
- torchvision >= 0.2.0
- matplotlib
- numpy
- opencv
- visdom (for visualization training process)
- wandb (for visualization training process)

NOTICE: different versions of PyTorch package have different memory usages.

## How to use
### Training on PASCAL VOC (20 classes)
```
main.py --mode train -data_path where/your/dataset/is --class_path ./names/VOC.names --num_class 20 --use_augmentation True --use_visdom True
```

### Test on PASCAL VOC (20 classes)
```
main.py  --mode test --data_path where/your/dataset/is --class_path ./names/VOC.names --num_class 20 --checkpoint_path your_checkpoint.pth.tar
```

## Supported Datasets
Only Pascal VOC datasets are supported for now.

## Configuration Options
|argument          |type|description|default|
|:-----------------|:----|:---------------------- |:----|
|--mode            |str  |train or test           |train|
|--dataset         |str  |only support voc now    |voc  |
|--data_path       |str  |data path               |     |
|--class_path      |str  |filenames text file path|     |
|--input_height    |int  |input height            |448  |
|--input_width     |int  |input width             |448  |
|--batch_size      |int  |batch size              |16   |
|--num_epochs      |int  |# of epochs             |16000|
|--learning_rate   |float|initial learning rate   |1e-3 |
|--dropout         |float|dropout probability     |0.5  |
|--num_gpus        |int  |# of GPUs for training  |1    |
|--checkpoint_path |str  |checkpoint path         |./   |
|--use_augmentation|bool |image Augmentation      |True |
|--use_visdom      |bool |visdom                  |False|
|--use_wandb       |bool |wandb                   |False|
|--use_summary     |bool |descripte Model summary |True |
|--use_gtcheck     |bool |gt check flag           |False|
|--use_githash     |bool |use githash             |False|
|--num_class       |int  |number of classes       |5    |

## Results 
Todo: Result Images here!!

## Authorship
This project is equally contributed by [Chanhee Jeong](https://github.com/chjeong530), [Donghyeon Hwang](https://github.com/ssaru), and [Jaewon Lee](https://github.com/insurgent92).

## Pre-trained models
Todo: model here!  

## Copyright
See [LICENSE](./LICENSE) for details.

## REFERENCES
[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. (https://arxiv.org/abs/1506.02640)
