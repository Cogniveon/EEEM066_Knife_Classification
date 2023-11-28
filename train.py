import argparse
import os
import signal
import warnings
from datetime import datetime
from timeit import default_timer as timer

import pandas as pd
import timm
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import config
from data import knifeDataset
from utils import AverageMeter, Logger, get_learning_rate, time_to_str

log = Logger()
should_exit = False


def init_logging(config):
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    log.open("logs/%s_log_train.txt" % (datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    log.write(
        "\n----------------------------------------------- [START %s] %s\n\n"
        % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-" * 51)
    )
    log.write("Using config values:\n")
    log.write("-" * 100)

    for k, v in config.__class__.__dict__.items():
        if not k.startswith("__"):
            log.write(f"{k}: {v}\n")

    log.write("                           |----- Train -----|----- Valid----|---------|\n")
    log.write("mode     iter     epoch    |       loss      |        mAP    | time    |\n")
    log.write("-------------------------------------------------------------------------------------------\n")


# Define a signal handler for the interrupt signal (Ctrl+C)
def interrupt_handler(signum, frame):
    # print(f"\nInterrupt signal received. {signum} , {frame} ; Exiting...")
    global should_exit
    should_exit = True


def train_model(model, loader, loss_fn, scaler, optimizer, epoch, validation_accuracy, start):
    losses = AverageMeter()
    model.train()
    model.training = True
    for i, (images, target, fnames) in enumerate(loader):
        img = images.cuda(non_blocking=True)
        label = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(img)
        loss = loss_fn(logits, label)
        losses.update(loss.item(), images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        print("\r", end="", flush=True)
        message = "%s %5.1f %6.1f        |      %0.3f     |      %0.3f     | %s" % (
            "train",
            i,
            epoch,
            losses.avg,
            validation_accuracy[0],
            time_to_str((timer() - start), "min"),
        )
        print(message, end="", flush=True)

    log.write("\n")
    log.write(message)

    return [losses.avg]


def evaluate_model(model, val_loader, loss_fn, epoch, train_loss, start):
    model.cuda()
    model.eval()
    model.training = False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))
            print("\r", end="", flush=True)
            message = "%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s" % (
                "val",
                i,
                epoch,
                train_loss[0],
                map.avg,
                time_to_str((timer() - start), "min"),
            )
            print(message, end="", flush=True)
        log.write("\n")
        log.write(message)
    return [map.avg]


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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGINT, interrupt_handler)

    parser = argparse.ArgumentParser(description="EEEM066 Knife classification coursework model trainer")

    # Get all available configs from the config.py module
    variable_names = [var for var in dir(config) if not callable(getattr(config, var)) and not var.startswith("__")]

    parser.add_argument(
        "--config", "-c", choices=variable_names, default=variable_names[0], help="Config to use for training"
    )

    args = parser.parse_args()

    # log.write(f"Training model using config: {args.config}")
    config = getattr(config, args.config)
    init_logging(config)

    train_loader = DataLoader(
        knifeDataset(pd.read_csv("dataset/train.csv"), config, mode="train"),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )

    val_loader = DataLoader(
        knifeDataset(pd.read_csv("dataset/val.csv"), config, mode="val"),
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )

    model = timm.create_model(config.base_model, pretrained=True, num_classes=config.n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.epochs * len(train_loader),
        eta_min=0,
        last_epoch=-1,
    )
    criterion = nn.CrossEntropyLoss().cuda()
    scaler = torch.cuda.amp.GradScaler()

    validation_metrics = [0]

    start = timer()
    for epoch in range(0, config.epochs):
        if should_exit:
            break

        lr = get_learning_rate(optimizer)

        training_metrics = train_model(
            model, train_loader, criterion, scaler, optimizer, epoch, validation_metrics, start
        )

        validation_metrics = evaluate_model(model, val_loader, criterion, epoch, training_metrics, start)

        filename = f"Knife-{config.base_model}-E" + str(epoch + 1) + ".pt"
        torch.save(model.state_dict(), filename)
