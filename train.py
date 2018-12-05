import os
import warnings

import git
import torch
import torchvision.transforms as transforms
import visdom
import wandb

import yolov1

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
# plt.ion()   # interactive mode
# model = torch.nn.DataParallel(net, device_ids=[0]).cuda()


def train(params):

    # future work variable
    dataset = params["dataset"]
    input_height = params["input_height"]
    input_width = params["input_width"]

    data_path = params["data_path"]
    class_path = params["class_path"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    learning_rate = params["lr"]
    dropout = params["dropout"]
    num_gpus = [i for i in range(params["num_gpus"])]
    checkpoint_path = params["checkpoint_path"]

    USE_VISDOM = params["use_visdom"]
    USE_WANDB = params["use_wandb"]
    USE_SUMMARY = params["use_summary"]
    USE_AUGMENTATION = params["use_augmentation"]
    USE_GTCHECKER = params["use_gtcheck"]

    USE_GITHASH = params["use_githash"]
    num_class = params["num_class"]

    if (USE_WANDB):
        wandb.init()
        wandb.config.update(params)  # adds all of the arguments as config variables

    with open(class_path) as f:
        class_list = f.read().splitlines()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if (USE_GITHASH):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        short_sha = repo.git.rev_parse(sha, short=7)

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
    net = yolov1.YOLOv1(params={"dropout": dropout, "num_class": num_class})
    # model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()

    print("device : ", device)
    if device.type == 'cpu':
        model = torch.nn.DataParallel(net)
    else:
        model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()


    if USE_SUMMARY:
        summary(model, (3, 448, 448))

    # 7.Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Train the model
    total_step = len(train_loader)

    total_train_step = num_epochs * total_step

    # for epoch in range(num_epochs):
    for epoch in range(1, num_epochs+1):

        if (epoch == 200) or (epoch == 400) or (epoch == 600) or (epoch == 20000) or (epoch == 30000):
            scheduler.step()

        for i, (images, labels, sizes) in enumerate(train_loader):

            current_train_step = (epoch) * total_step + (i + 1)

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
            objness1_loss = detection_loss_4_yolo(outputs, labels, device.type)
            # objness1_loss = detection_loss_4_yolo(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (((current_train_step) % 100) == 0) or (current_train_step % 10 == 0 and current_train_step < 100):
                print(
                    'epoch: [{}/{}], total step: [{}/{}], batch step [{}/{}], lr: {}, total_loss: {:.4f}, coord1: {:.4f}, size1: {:.4f}, noobj_clss: {:.4f}, objness1: {:.4f}, class_loss: {:.4f}'
                    .format(epoch + 1, num_epochs, current_train_step, total_train_step, i + 1, total_step,
                            ([param_group['lr'] for param_group in optimizer.param_groups])[0],
                            loss.item(), obj_coord1_loss, obj_size1_loss, noobjness1_loss, objness1_loss, obj_class_loss))

                if USE_VISDOM:
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), loss.item(), iter_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), obj_coord1_loss, coord1_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), obj_size1_loss, size1_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), obj_class_loss, obj_cls_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), noobjness1_loss, noobjectness1_plot, None, 'append')
                    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), objness1_loss, objectness1_plot, None, 'append')

                if USE_WANDB:
                    wandb.log({'total_loss': loss.item(), 'obj_coord1_loss': obj_coord1_loss, 'obj_size1_loss': obj_size1_loss,
                            'obj_class_loss': obj_class_loss, 'noobjness1_loss': noobjness1_loss, 'objness1_loss': objness1_loss})

        if not USE_GITHASH:
            short_sha = 'noHash'

        # if ((epoch % 1000) == 0) and (epoch != 0):
        if ((epoch % 1000) == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': "YOLOv1",
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False, filename=os.path.join(checkpoint_path, 'ckpt_{}_ep{:05d}_loss{:.04f}_lr{}.pth.tar'.format(short_sha, epoch, loss.item(), ([param_group['lr'] for param_group in optimizer.param_groups])[0])))

