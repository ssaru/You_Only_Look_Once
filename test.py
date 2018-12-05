import os
import torch
import yolov1
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from torchsummary.torchsummary import summary
from PIL import Image, ImageDraw


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

    objness_threshold = 0.15

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = yolov1.YOLOv1(params={"dropout": 1.0, "num_class": num_class})
    # model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()
    print("device : ", device)
    if device is "cpu":
        model = torch.nn.DataParallel(net)
    else:
        model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()

    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()

    if USE_SUMMARY:
        summary(model, (3, 448, 448))

    image_path = os.path.join(data_path, "JPEGImages")
    root, dir, files = next(os.walk(os.path.abspath(image_path)))

    for file in files:
        extension = file.split(".")[-1]
        if extension not in ["jpeg", "jpg", "png", "JPEG", "JPG", "PNG"]:
            continue

        img = Image.open(os.path.join(image_path, file)).convert('RGB')

        # PRE-PROCESSING
        input_img = img.resize((input_width, input_height))
        input_img = transforms.ToTensor()(input_img)
        c, w, h = input_img.shape

        # INVERSE TRANSFORM IMAGE########
        # inverseTimg = transforms.ToPILImage()(input_img)
        W, H = img.size
        draw = ImageDraw.Draw(img)

        dx = W // 7
        dy = H // 7
        ##################################

        input_img = input_img.view(1, c, w, h)
        input_img = input_img.to(device)

        # INFERENCE
        outputs = model(input_img)
        b, w, h, c = outputs.shape

        outputs = outputs.view(w, h, c)
        outputs_np = outputs.cpu().data.numpy()

        outputs[:, :, 0] = torch.sigmoid(outputs[:, :, 0])
        outputs[:, :, 5:] = torch.sigmoid(outputs[:, :, 5:])

        objness = outputs[:, :, 0].cpu().data.numpy()

        print("OBJECTNESS : {}".format(objness.shape))
        print(objness)
        print("\n\n\n")
        print("IMAGE SIZE")
        print("width : {}, height : {}".format(W, H))
        print("\n\n\n\n")

        try:

            for i in range(7):
                for j in range(7):
                    draw.rectangle(((dx * i, dy * j), (dx * i + dx, dy * j + dy)), outline='#00ff88')

                    if objness[i][j] >= objness_threshold:
                        block = outputs_np[i][j]

                        x_start_point = dx * i
                        y_start_point = dy * j

                        x_shift = block[1]
                        y_shift = block[2]

                        center_x = int((block[1] * W / 7.0) + (i * W / 7.0))
                        center_y = int((block[2] * H / 7.0) + (j * H / 7.0))
                        w_ratio = block[3]
                        h_ratio = block[4]
                        w_ratio = w_ratio * w_ratio
                        h_ratio = h_ratio * h_ratio
                        width = int(w_ratio * W)
                        height = int(h_ratio * H)

                        xmin = center_x - (width // 2)
                        ymin = center_y - (height // 2)
                        xmax = xmin + width
                        ymax = ymin + height

                        clsprob = block[5:]
                        cls_idx = np.argmax(clsprob)

                        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue")
                        draw.text((xmin, ymin), class_list[cls_idx])
                        draw.ellipse(((center_x - 2, center_y - 2),
                                      (center_x + 2, center_y + 2)),
                                     fill='blue')

                        # LOG
                        print("idx : [{}][{}]".format(i, j))
                        print("x shift : {}, y shift : {}".format(x_shift, y_shift))
                        print("w ratio : {}, h ratio : {}".format(w_ratio, h_ratio))
                        print("cls prob : {}".format(np.around(clsprob, decimals=2)))

                        print("xmin : {}, ymin : {}, xmax : {}, ymax : {}".format(xmin, ymin, xmax, ymax))
                        print("width : {} height : {}".format(width, height))
                        print("class list : {}".format(class_list))
                        print("\n\n\n")

            plt.figure(figsize=(24, 18))
            plt.imshow(img)
            plt.show()
            plt.close()

        except Exception as e:
            print("ERROR")
            print("Message : {}".format(e))
