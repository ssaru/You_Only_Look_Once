import os
import torch
import yolov1
import matplotlib.pyplot as plt

from torchvision import transforms
from torchsummary.torchsummary import summary
from PIL import Image

def test(params):

    input_height = params["input_height"]
    input_width = params["input_width"]

    data_path = params["data_path"]
    class_path = params["class_path"]
    num_gpus = [i for i in range(params["num_gpus"])]
    checkpoint_path = params["checkpoint_path"]

    USE_SUMMARY = params["use_summary"]

    num_class = params["num_class"]

    with open(class_path) as f:
        class_list = f.read().splitlines()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = yolov1.YOLOv1(params={"dropout": 1.0, "num_class": num_class})
    model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()


    if USE_SUMMARY:
        summary(model, (3, 448, 448))

    root, dir, files = next(os.walk(os.path.abspath(data_path)))

    for file in files:
        extension = file.split(".")[-1]
        if extension not in ["jpeg", "jpg", "png", "JPEG","JPG", "PNG"]:
            continue

        img = Image.open(os.path.join(data_path, file)).convert('RGB')
        plt.imshow(img)
        plt.show()
        plt.close()
        img_size = img.size
        img = img.resize((input_width, input_height))
        img = transforms.ToTensor()(img)
        c, w, h = img.shape
        img = img.view(1, c, w, h)

        img = img.to(device)

        outputs = model(img)
        b, w, h, c = outputs.shape

        outputs = outputs.view(w, h, c)
        print(outputs)
        print(outputs.shape)

        objness = outputs[:, :, 0].cpu().data.numpy()
        x_shift = outputs[:, :, 1].cpu().data.numpy()
        y_shift = outputs[:, :, 2].cpu().data.numpy()
        w_ratio = outputs[:, :, 3].cpu().data.numpy()
        h_ratio = outputs[:, :, 4].cpu().data.numpy()
        clsprob = outputs[:, :, 5:].cpu().data.numpy()

        _, _, c = clsprob.shape

        """
        for i in range(c):
            clsprob[:,:,i] = objness * clsprob[:,:,i]
        """
        print(objness.shape)
        print(x_shift.shape)
        print(y_shift.shape)
        print(w_ratio.shape)
        print(h_ratio.shape)
        print(clsprob.shape)

        objness[objness > 0.1] = 1
        objness[objness <= 0.1] = 0

        print()
        print("OBJECTNESS")
        print()
        print(outputs[:, :, 0])
        print()
        print(objness)
        print()
        print("TOTAL PROB")
        print()
        print(clsprob[:, :, 0])
        print()
        print(clsprob[:, :, 1])
        print()
        print(clsprob[:, :, 2])
        print()
        print(clsprob[:, :, 3])
        print()
        print(clsprob[:, :, 4])
        exit()


    pass
