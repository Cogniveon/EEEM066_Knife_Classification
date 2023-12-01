import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import *


class KnifeDataset(Dataset):
    def __init__(self, parquet_path, config, mode="train"):
        self.parquet_path = parquet_path
        self.config = config
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
                    T.Resize((self.config.img_width, self.config.img_height)),
                    T.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
                    T.RandomRotation(degrees=(0, 180)),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )(X)
        elif self.mode == "val":
            X = T.Compose(
                [
                    T.Resize((self.config.img_width, self.config.img_height)),
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
