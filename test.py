## import libraries for training
import argparse
import warnings

import pandas as pd
import timm
import torch.optim
from torch.utils.data import DataLoader

import config
from data import knifeDataset
from utils import *


# Validating the model
def evaluate(val_loader, model):
    model.cuda()
    model.eval()
    model.training = False
    map = AverageMeter()
    acc1 = AverageMeter()
    acc5 = AverageMeter()
    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))
            acc1.update(valid_acc1, img.size(0))
            acc5.update(valid_acc5, img.size(0))
    return map.avg, acc1.avg, acc5.avg


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


def is_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{path} is not a valid path")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="EEEM066 Knife classification coursework model trainer")

    # Get all available configs from the config.py module
    variable_names = [var for var in dir(config) if not callable(getattr(config, var)) and not var.startswith("__")]

    parser.add_argument(
        "--config", "-c", choices=variable_names, default=variable_names[0], help="Config to use for training"
    )
    parser.add_argument("--model", "-m", type=is_path, required=True, help="Model to be tested")

    args = parser.parse_args()

    # log.write(f"Training model using config: {args.config}")
    config = getattr(config, args.config)

    ######################## load file and get splits #############################
    print("reading test file")
    # Expected csv to have 2 columns: Id,Label (Eg: ./dataset/test/Anglo_Arms_95_Fixed_Blade-b.png,2)
    test_files = pd.read_csv("dataset/test.csv")
    print("Creating test dataloader")
    test_gen = knifeDataset(test_files, config, mode="val")
    test_loader = DataLoader(test_gen, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("loading trained model")
    # model = timm.create_model("tf_efficientnet_b0", pretrained=True, num_classes=config.n_classes)
    model = config.base_model
    model.load_state_dict(torch.load(args.model))
    model.to(device)

    ############################# Training #################################
    print("Evaluating trained model")
    map, acc1, acc5 = evaluate(test_loader, model)
    print("mAP   =", map.item())
    print("acc@1 =", acc1.item())
    print("acc@5 =", acc5.item())
