import os
import torch
import yolov1

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
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    transform = transforms.ToTensor()

    if USE_SUMMARY:
        summary(model, (3, 448, 448))

    root, dir, files = next(os.walk(os.path.abspath(data_path)))

    for file in files:
        extension = file.split(".")[-1]
        if extension not in ["jpeg", "jpg", "png", "JPEG","JPG", "PNG"]:
            continue

        img = Image.open(os.path.join(data_path, file)).convert('RGB')
        img_size = img.size
        img = img.resize((input_width, input_height))
        img = transform(img)

        print(img)
        print(type(img))
        print(img.shape)
        exit()

        img = img.to(device)

        outputs = model(img)
        print(outputs)


    pass
