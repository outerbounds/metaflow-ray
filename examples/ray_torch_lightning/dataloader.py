import os
import torch
import lightning as L
from torchvision import transforms
from typing import Optional, Literal
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = "~/.datasets/mnist", batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # Downloading dataset; executed only on rank 0 in distributed setups
        MNIST(self.data_dir, train=True, download=True)

    def setup(self, stage: Optional[Literal["fit"]] = None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
