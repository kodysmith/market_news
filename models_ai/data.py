import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(config):
    """Return train_loader, val_loader, test_loader for the configured dataset."""
    if config.data.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        full_train = datasets.MNIST(
            config.data.data_dir, train=True, download=True, transform=transform
        )
        test_set = datasets.MNIST(
            config.data.data_dir, train=False, download=True, transform=transform
        )
        input_dim = 784
        output_dim = 10
    elif config.data.dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
        ])
        full_train = datasets.EMNIST(
            config.data.data_dir, split="balanced", train=True,
            download=True, transform=transform,
        )
        test_set = datasets.EMNIST(
            config.data.data_dir, split="balanced", train=False,
            download=True, transform=transform,
        )
        input_dim = 784
        output_dim = 47
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    # Split train into train + val
    val_size = int(len(full_train) * config.data.val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    loader_kwargs = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, input_dim, output_dim
