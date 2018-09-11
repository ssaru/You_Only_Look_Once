import numpy as np
import torch
num_classes = 1


def detection_collate_test(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    targets = []
    imgs = []
    sizes = []
    #print(len(batch[0]))
    for sample in batch:
        imgs.append(sample[0])
        #sizes.append(sample[2])
        
        #Add Size
        #shape_np = sample[2]
        #shape = torch.Tensor(shape_np)
        sizes.append(sample[2])
        
        np_label = np.zeros((7,7,6), dtype=np.float32)
        for i in range(7):
            for j in range(7):
                np_label[i][j][1] = (num_classes-1)
        
        for object in sample[1]:
            objectness=1
            #print(len(object))
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

    return torch.stack(imgs,0), torch.stack(targets, 0), sizes


# visdom function

def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 1)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(viz, iteration, loss, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 1)).cpu() * iteration,
        Y=torch.Tensor([loss]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def one_hot(output , label):

    label = label.cpu().data.numpy()
    b, s1, s2, c = output.shape
    dst = np.zeros([b,s1,s2,c], dtype=np.float32)

    for k in range(b):
        for i in range(s1):
            for j in range(s2):

                dst[k][i][j][int(label[k][i][j])] = 1.

    return torch.from_numpy(dst)