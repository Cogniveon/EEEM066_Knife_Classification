import argparse
import os
import warnings
from datetime import datetime
from timeit import default_timer as timer

import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import config
from data import knifeDataset
from utils import AverageMeter, Logger, get_learning_rate, time_to_str

log = Logger()


def init_logging(output_path, config):
    log.open(f"{output_path}/log.txt")
    log.write(
        "\n----------------------------------------------- [START %s] %s\n\n"
        % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-" * 51)
    )

    log.write("Using config values:\n\n")
    for key in config.__dict__:
        if not key.startswith("__") and key != "base_model":
            log.write(f"{key}: {getattr(config, key)}\n")
    log.write("-" * 127 + "\n")

    log.write("                           |----- Train -----|----- Valid----|---------|\n")
    log.write("mode     iter     epoch    |       loss      |        mAP    | time    |\n")
    log.write("-------------------------------------------------------------------------------------------\n")


def train_model(model, loader, loss_fn, scaler, optimizer, epoch, validation_accuracy, start):
    losses = AverageMeter()
    model.train()
    model.training = True
    for i, (images, target, fnames) in enumerate(loader):
        img = images.cuda(non_blocking=True)
        label = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(img)
            loss = loss_fn(logits, label)

        losses.update(loss.item(), images.size(0))

        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

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


def evaluate_model(model, val_loader, epoch, train_loss, start):
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

    parser = argparse.ArgumentParser(description="EEEM066 Knife classification coursework model trainer")

    # Get all available configs from the config.py module
    variable_names = [var for var in dir(config) if not callable(getattr(config, var)) and not var.startswith("__")]

    parser.add_argument(
        "--config", "-c", choices=variable_names, default=variable_names[0], help="Config to use for training"
    )

    args = parser.parse_args()

    # log.write(f"Training model using config: {args.config}")
    config = getattr(config, args.config)

    output_path = f"./results/{config.model_name}/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    init_logging(output_path, config)

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

    model = config.base_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

    # scheduler = lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer,
    #     T_max=config.epochs * len(train_loader),
    #     eta_min=0,
    #     last_epoch=-1,
    # )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # criterion = nn.BCEWithLogitsLoss(torch.ones([config.n_classes])).to(device)
    # criterion = nn.SmoothL1Loss().to(device)
    criterion = nn.CrossEntropyLoss()

    # torch.autograd.set_detect_anomaly(True)
    scaler = torch.cuda.amp.GradScaler()

    validation_metrics = [0]

    start = timer()
    for epoch in range(0, config.epochs):
        lr = get_learning_rate(optimizer)

        training_metrics = train_model(
            model, train_loader, criterion, scaler, optimizer, epoch, validation_metrics, start
        )

        validation_metrics = evaluate_model(model, val_loader, epoch, training_metrics, start)

        scheduler.step()

        filename = f"{output_path}/Knife-{config.model_name}-E" + str(epoch + 1) + ".pt"
        torch.save(model.state_dict(), filename)
