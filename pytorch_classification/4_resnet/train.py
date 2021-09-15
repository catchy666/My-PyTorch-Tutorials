import json
import os

import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torchvision.models import resnet34
from tqdm import tqdm

from model import resnet50

# configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NW = 0
# NW = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])
NC = 5


def train(train_loader, val_loader):
    net = resnet50(pretrained=True, num_classes=5)
    net.to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    epochs = 10
    model_path = "./resnet-e{}.pth".format(epochs)
    best_acc = 0.0

    # training
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(DEVICE))
            loss = loss_function(logits, labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # eval
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in tqdm(val_loader):
                val_images, val_labels = val_data
                outputs = net(val_images.to(DEVICE))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(DEVICE)).sum().item()

        val_acc = acc / len(val_loader.dataset)
        print("[epoch %d] train_loss: %.3f  val_acc: %.3f " % (
            epoch + 1, running_loss / len(train_loader.dataset), val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), model_path)

    print('Finished Training')


def main():
    print("using device: %s" % DEVICE)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = data_root + "/data_set/flower_data/"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(os.path.join(image_path, "train"),
                                         data_transform["train"])
    val_dataset = datasets.ImageFolder(os.path.join(image_path, "val"),
                                       data_transform["val"])

    flower_list = train_dataset.class_to_idx
    class_dict = dict((val, key) for key, val in flower_list.items())
    # print(class_dict)
    with open("class_indices.json", "w") as f:
        f.write(json.dumps(class_dict, indent=4))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=NW)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=NW)
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))

    train(train_loader, val_loader)


if __name__ == '__main__':
    main()
