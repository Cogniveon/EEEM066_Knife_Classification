import os
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from utils import AverageMeter, Logger, time_to_str


class KnifeDataset(Dataset):
    def __init__(self, parquet_path, mode="train"):
        self.parquet_path = parquet_path
        self.mode = mode

        # Load metadata from Parquet file
        self.images_df = pd.read_parquet(self.parquet_path)

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        X, fname = self.read_images(index)
        labels = self.images_df.iloc[index].Label
        if self.mode == "train":
            X = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )(X)
        elif self.mode == "val":
            X = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )(X)
        return X.float(), labels, fname

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id)
        # im = cv2.imread(filename)[:, :, ::-1]
        # return im, filename
        with open(filename, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB"), filename


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.conv2_bn = nn.BatchNorm2d(128)

        # self.relu2 = nn.PReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.conv3_bn = nn.BatchNorm2d(256)

        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # for 2 conv layers, 128 * 56 * 56, 512
        # for 3 conv layers, 256 * 28 * 28, 512
        # self.fc1 = nn.Linear(64 * 112 * 112, 512)
        # self.fc1_bn = nn.BatchNorm1d(512)

        # self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64 * 112 * 112, num_classes)

        # Initialize weights using Xavier initialization
        init.xavier_uniform_(self.conv1.weight)
        # init.xavier_uniform_(self.conv2.weight)
        # init.xavier_uniform_(self.conv3.weight)
        # init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.pool1(x)
        x = self.relu1(x)

        # x = self.conv2(x)
        # x = self.conv2_bn(x)
        # x = self.pool2(x)
        # x = self.relu2(x)

        # x = self.conv3(x)
        # x = self.conv3_bn(x)
        # x = self.pool3(x)
        # x = self.relu3(x)

        x = self.flatten(x)
        # x = self.relu4(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x


## Computing the mean average precision, accuracy
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [
            correct[0],
            correct[0] + correct[1] + correct[2] + correct[3] + correct[4],
        ]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5


batch_size = 8
learning_rate = 0.01
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_loader = DataLoader(
    KnifeDataset("dataset/train.parquet"),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=os.cpu_count(),
)

val_loader = DataLoader(
    KnifeDataset("dataset/val.parquet", mode="val"),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=os.cpu_count(),
)


model = CustomCNN(num_classes=192).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=num_epochs * len(train_loader),
    eta_min=0,
    last_epoch=-1,
)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)  # Adjust parameters as needed
criterion = nn.CrossEntropyLoss().to(device)
# criterion = FocalLoss(alpha=0.75, gamma=2, reduction="mean").to(device)


# GradScaler for mixed precision training
torch.autograd.set_detect_anomaly(True)
scaler = GradScaler()


output_path = f"./results/customcnn/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
if not os.path.exists(output_path):
    os.makedirs(output_path)

log = Logger()
log.open(f"{output_path}/log.txt")
log.write(
    "\n---------------------------------------- [START %s] %s\n\n"
    % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-" * 51)
)

log.write("Using config values:\n\n")
log.write(f"n_classes: 192\n")
log.write(f"model_name: customcnn\n")
log.write(f"{model}\n")
log.write(f"img_width: 224\n")
log.write(f"img_height: 224\n")
log.write(f"batch_size: {batch_size}\n")
log.write(f"epochs: {num_epochs}\n")
log.write(f"learning_rate: {learning_rate}\n")
log.write(f"device: {device}\n")
log.write(f"lr_scheduler: {scheduler.__class__.__name__}\n")
log.write(f"loss_fn: {criterion.__class__.__name__}\n")
log.write(f"optimizer: {optimizer.__class__.__name__}\n")
log.write("-" * 120 + "\n")

log.write("                           |----- Train -----|----- Valid----|---------------|---------------|----------|\n")
log.write("mode     iter     epoch    |       loss      |        mAP    |      acc@1    |      acc@5    | time     |\n")
log.write("-" * 120 + "\n")


validation_accuracy = [[torch.tensor(0), torch.tensor(0), torch.tensor(0)]]
start = timer()

for epoch in range(num_epochs):
    model.train()

    losses = AverageMeter()
    for batch_idx, (imgs, labels, fnames) in enumerate(train_loader):
        images, labels = imgs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        del images, labels, outputs
        torch.cuda.empty_cache()
        losses.update(loss.item(), imgs.size(0))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        # Backward pass
        scaler.scale(loss).backward()
        # After the backward pass, before the optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        scaler.step(optimizer)
        scaler.update()

        print("\r", end="", flush=True)
        last_map, last_acc1, last_acc5 = validation_accuracy[-1]
        message = "%s   %5.1f %6.1f       |      %0.3f     |     %0.3f    |     %0.3f    |     %0.3f    | %s" % (
            "train",
            batch_idx,
            epoch,
            losses.avg,
            last_map,
            last_acc1,
            last_acc5,
            time_to_str((timer() - start), "min"),
        )
        print(message, end="", flush=True)

    log.write("\n")
    log.write(message)

    model.eval()
    model.training = False
    map = AverageMeter()
    acc1 = AverageMeter()
    acc5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (imgs, labels, fnames) in enumerate(val_loader):
            images, labels = imgs.to(device), labels.to(device)

            with autocast():
                logits = model(images)
                preds = logits.softmax(1)

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, labels)
            map.update(valid_map5, imgs.size(0))
            acc1.update(valid_acc1, imgs.size(0))
            acc5.update(valid_acc5, imgs.size(0))
            print("\r", end="", flush=True)
            message = "%s   %5.1f %6.1f       |      %0.3f     |     %0.3f    |     %0.3f    |     %0.3f    | %s" % (
                "val  ",
                batch_idx,
                epoch,
                losses.avg,
                map.avg,
                acc1.avg,
                acc5.avg,
                time_to_str((timer() - start), "min"),
            )
            print(message, end="", flush=True)
        log.write("\n")
        log.write(message)

        validation_accuracy.append([map.avg, acc1.avg, acc5.avg])

    # Update learning rate scheduler
    scheduler.step()

    filename = f"{output_path}/Knife-customcnn-E" + str(epoch + 1) + ".pt"
    torch.save(model.state_dict(), filename)


mAP_values = ["%0.3f" % (mAP.item()) for i, (mAP, _, _) in enumerate(validation_accuracy)]
print(f"\n\nmAP = [{' '.join(mAP_values)}];")

best_epoch, best_mAP = max([(i, mAP.item()) for i, (mAP, _, _) in enumerate(validation_accuracy)], key=lambda x: x[1])

print(f"\nbest mAP = {best_mAP} at epoch {best_epoch + 1}")
