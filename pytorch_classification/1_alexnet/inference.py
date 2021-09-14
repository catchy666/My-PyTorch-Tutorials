import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import AlexNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "./alexNet-e10.pth"


def main():
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./test.jpg"
    assert os.path.exists(img_path), "file : '{}' does not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)

    img = data_transform(img)
    img = torch.unsqueeze(img, 0)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    class_dict = json.load(open(json_path, 'r'))

    # create model
    net = AlexNet(num_classes=5).to(DEVICE)

    # load weights
    assert os.path.exists(CHECKPOINT), "file: '{}' does not exist.".format(CHECKPOINT)
    net.load_state_dict(torch.load(CHECKPOINT))

    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(img.to(DEVICE))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    res_info = "class: {} prob: {:.3}".format(class_dict[str(predict_cla)],
                                              predict[predict_cla].numpy())
    plt.title(res_info)
    print(res_info)
    plt.show()


if __name__ == '__main__':
    main()
