import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ray import tune
from ray.air import session, RunConfig
from ray.tune.schedulers import ASHAScheduler

from functools import partial


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, data_loader, batch_size=512, smoke_test=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        # exit early only on a small subset of data
        if smoke_test and batch_idx * len(data) > batch_size:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, batch_size=256, smoke_test=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # exit early only on a small subset of data
            if smoke_test and batch_idx * len(data) > batch_size:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / total


def train_mnist(
    config, batch_size, test_batch_size, smoke_test=True, n_epochs=10
):
    mnist_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(
        datasets.MNIST("./", train=True, download=True, transform=mnist_transforms),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./", train=False, download=True, transform=mnist_transforms),
        batch_size=test_batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    for i in range(n_epochs):
        train(model, optimizer, train_loader, batch_size, smoke_test=smoke_test)
        acc = test(model, test_loader, test_batch_size, smoke_test=smoke_test)
        session.report({"mean_accuracy": acc})
        if i % 5 == 0:
            torch.save(model.state_dict(), f"./model_{i}.pth")


def run(
    num_cpus,
    num_gpus,
    search_space=None,
    batch_size=512,
    test_batch_size=256,
    num_samples=20,
    smoke_test=True,
    run_config_storage_path=None,
    n_epochs=10,
):
    trainable = partial(
        train_mnist,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        smoke_test=smoke_test,
        n_epochs=n_epochs,
    )
    trainable_with_gpu = tune.with_resources(trainable, {"cpu": num_cpus, "gpu": num_gpus})
    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
        ),
        param_space=search_space,
        run_config=None if run_config_storage_path is None else RunConfig(storage_path=run_config_storage_path),
    )
    results = tuner.fit()
    return results


def plot(results, ax=None):
    dfs = {result.path: result.metrics_dataframe for result in results}
    for d in dfs.values():
        try:
            ax = d.mean_accuracy.plot(ax=ax, legend=False)
        except AttributeError:
            pass
    return dfs
