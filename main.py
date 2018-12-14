# -*- coding:utf-8 -*-

import argparse

from train import train
from test import test

parser = argparse.ArgumentParser(description='YOLO v1.')
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--dataset', type=str, help='dataset to train on, voc', default='voc')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--class_path', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=448)
parser.add_argument('--input_width', type=int, help='input width', default=448)
parser.add_argument('--batch_size', type=int, help='batch size', default=16)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=16000)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--dropout', type=float, help='dropout probability', default=0.5)
parser.add_argument('--num_gpus', type=int, help='number of GPUs to use for training', default=1)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='./')

# flag
parser.add_argument('--use_augmentation', type=lambda x: (str(x).lower() == 'true'), help='Image Augmentation', default=True)
parser.add_argument('--use_visdom', type=lambda x: (str(x).lower() == 'true'), help='visdom board', default=False)
parser.add_argument('--use_wandb', type=lambda x: (str(x).lower() == 'true'), help='wandb', default=False)
parser.add_argument('--use_summary', type=lambda x: (str(x).lower() == 'true'), help='descripte Model summary', default=True)
parser.add_argument('--use_gtcheck', type=lambda x: (str(x).lower() == 'true'), help='Ground Truth check flag', default=False)
parser.add_argument('--use_githash', type=lambda x: (str(x).lower() == 'true'), help='use githash to checkpoint', default=False)

# develop
parser.add_argument('--num_class', type=int, help='number of class', default=5, required=True)
args = parser.parse_args()


def main():
    params = {
        "mode": args.mode,
        "dataset": args.dataset,
        "data_path": args.data_path,
        "class_path": args.class_path,
        "input_height": args.input_height,
        "input_width": args.input_width,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.learning_rate,
        "dropout": args.dropout,
        "num_gpus": args.num_gpus,
        "checkpoint_path": args.checkpoint_path,

        "use_visdom": args.use_visdom,
        "use_wandb": args.use_wandb,
        "use_summary": args.use_summary,
        "use_augmentation": args.use_augmentation,

        "num_class": args.num_class,
        "use_gtcheck": args.use_gtcheck,
        "use_githash": args.use_githash
    }

    if params["mode"] == "train":
        train(params)
    elif params["mode"] == "test":
        test(params)


if __name__ == '__main__':
    main()
