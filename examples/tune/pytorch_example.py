import numpy as np
import subprocess
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import ScalingConfig
from ray.air import RunConfig

from functools import partial


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # In this example, we don't change the model architecture
        # due to simplicity.
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, train_loader, epoch_size=512, smoke_test=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if smoke_test and batch_idx * len(data) > epoch_size:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, test_size=256, smoke_test=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if smoke_test and batch_idx * len(data) > test_size:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def download_mnist():
    subprocess.run(
        ["wget", "www.di.ens.fr/~lelarge/MNIST.tar.gz"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        ["tar", "-zxvf", "MNIST.tar.gz"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def train_mnist(
    config, epoch_size, test_size, dataset="mnist", smoke_test=True, n_epochs=10
):
    # Data Setup
    if dataset == "mnist":
        try:
            download_mnist()
            mnist_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            train_loader = DataLoader(
                datasets.MNIST(
                    "./", train=True, download=True, transform=mnist_transforms
                ),
                batch_size=64,
                shuffle=True,
            )
            test_loader = DataLoader(
                datasets.MNIST(
                    "./", train=False, download=True, transform=mnist_transforms
                ),
                batch_size=64,
                shuffle=True,
            )
        except (ValueError, RuntimeError, EOFError) as e:
            print("Issue with MNIST dataset path.")
            session.report({"mean_accuracy": 0.0})
            return
    else:
        raise ValueError("Only MNIST dataset is currently supported by this interface.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    for i in range(n_epochs):
        train(model, optimizer, train_loader, epoch_size, smoke_test=smoke_test)
        acc = test(model, test_loader, test_size, smoke_test=smoke_test)

        # Send the current training result back to Tune
        session.report({"mean_accuracy": acc})

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")


def run(
    dataset="mnist",
    search_space=None,
    epoch_size=512,
    test_size=256,
    num_samples=20,
    smoke_test=True,
    run_config_storage_path=None,
    n_epochs=10,
):
    tuner = tune.Tuner(
        partial(
            train_mnist,
            dataset=dataset,
            test_size=test_size,
            smoke_test=smoke_test,
            n_epochs=n_epochs,
            epoch_size=epoch_size,
        ),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
        ),
        param_space=search_space,
        run_config=None
        if run_config_storage_path is None
        else RunConfig(storage_path=run_config_storage_path),
    )
    results = tuner.fit()
    return results


def plot(results, ax=None):
    dfs = {result.log_dir: result.metrics_dataframe for result in results}
    for d in dfs.values():
        try:
            ax = d.mean_accuracy.plot(ax=ax, legend=False)
        except AttributeError:
            pass
    return dfs


if __name__ == "__main__":
    search_space = {
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "momentum": tune.uniform(0.1, 0.9),
        # "scaling_config": ScalingConfig(resources_per_worker={"CPU": 1})
    }

    dfs = plot(run(search_space=search_space, smoke_test=True))
