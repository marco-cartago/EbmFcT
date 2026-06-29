import torch

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
from disentanglement_datasets import DSprites
from sklearn.datasets import fetch_olivetti_faces
import numpy as np



def load_fashion_mnist(batch_size=64, shuffle=True, class_subset=None):
    """Load FashionMNIST dataset with optional class subset filtering.
    Args:
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the training data.
        class_subset (list, optional): List of class labels to include. If None, includes all classes.
    Returns:
        train_loader, test_loader: Data loaders for training and testing datasets.
    Info:
        0 T-shirt/top
        1 Trouser
        2 Pullover
        3 Dress
        4 Coat
        5 Sandal
        6 Shirt
        7 Sneaker
        8 Bag
        9 Ankle boot
            
    
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_set = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    if class_subset is not None:
        # Filter indices where label is in the specified class subset
        train_idx = [i for i in range(len(train_set)) if train_set[i][1] in class_subset]
        test_idx = [i for i in range(len(test_set)) if test_set[i][1] in class_subset]

        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader

def load_mnist_digit(batch_size=64, shuffle=True, digit=None):
    """Load MNIST dataset with optional digit filtering.
    Args:
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the training data.
        digit (int, optional): Digit label to include. If None, includes all digits.
    Returns:
        train_loader, test_loader: Data loaders for training and testing datasets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_set = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    test_set = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    if digit is not None:
        # Filter indices where label is the specified digit
        train_idx = [i for i in range(len(train_set)) if train_set[i][1] == digit]
        test_idx = [i for i in range(len(test_set)) if test_set[i][1] == digit]

        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# Struttura latenti dSprites:
# idx 0 = color    (1 valore,  sempre 1)
# idx 1 = shape    (3 valori:  0=square, 1=ellipse, 2=heart)
# idx 2 = scale    (6 valori)
# idx 3 = orient   (40 valori)
# idx 4 = pos_x    (32 valori)
# idx 5 = pos_y    (32 valori)

DSPRITES_PATH = "./data/DSprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

class DSpritesDataset(Dataset):
    LATENT_NAMES = ["color", "shape", "scale", "orient", "pos_x", "pos_y"]
    LATENT_SIZES = [1, 3, 6, 40, 32, 32]
    SHAPE_NAMES  = {0: "square", 1: "ellipse", 2: "heart"}

    def __init__(self, path=DSPRITES_PATH, shape_filter=None, transform=None):
        data = np.load(path, encoding="latin1", allow_pickle=True)

        imgs    = data["imgs"]                          # uint8, (N, 64, 64) — ~700MB
        latents = data["latents_classes"]               # int64, (N, 6)      — trascurabile

        if shape_filter is not None:
            mask    = latents[:, 1] == shape_filter
            imgs    = imgs[mask]
            latents = latents[mask]

        # Tieni tutto in uint8: 1 byte/pixel invece di 4 (float32)
        # np.load con mmap_mode="r" evita anche di caricare in RAM tutto in una volta
        self.images  = imgs                             # uint8 numpy array, lazy
        self.latents = torch.from_numpy(latents.astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Conversione float32 + normalizzazione solo quando serve
        img = self.images[idx].astype(np.float32)      # (64, 64)
        img = torch.from_numpy(img).unsqueeze(0)       # (1, 64, 64)
        img = img * 2.0 - 1.0                          # [-1, 1]

        if self.transform is not None:
            img = self.transform(img)

        return {"image": img, "latents": self.latents[idx]}

def load_dsprites(
    path=DSPRITES_PATH,
    batch_size=64,
    shape_filter=None,       # 0=square | 1=ellipse | 2=heart | None=tutto
    train_split=0.8,
    shuffle_train=True,
    transform=None,
    seed=42,
):
    dataset = DSpritesDataset(path=path, shape_filter=shape_filter, transform=transform)

    train_size = int(train_split * len(dataset))
    test_size  = len(dataset) - train_size
    train_set, test_set = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,           num_workers=0)

    print(f"dSprites caricato — train: {len(train_set):,}  test: {len(test_set):,}  "
          f"(shape_filter={DSpritesDataset.SHAPE_NAMES.get(shape_filter, 'all')})")

    return train_loader, test_loader



def load_CIFAR10(batch_size=64, shuffle=True, class_subset=None):
    """Load FashionMNIST dataset with optional class subset filtering.
    Args:
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the training data.
        class_subset (list, optional): List of class labels to include. If None, includes all classes.
    Returns:
        train_loader, test_loader: Data loaders for training and testing datasets.
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    if class_subset is not None:
        # Filter indices where label is in the specified class subset
        train_idx = [i for i in range(len(train_set)) if train_set[i][1] in class_subset]
        test_idx = [i for i in range(len(test_set)) if test_set[i][1] in class_subset]

        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader


def load_olivetti(batch_size: int = 64, shuffle: bool = True):
    """
    Load the Olivetti faces dataset.

    Args:
        batch_size (int): Batch size for the loaders.
        shuffle (bool): Shuffle the training set.
    
    Returns:
        train_loader, test_loader: DataLoaders for train / test splits.
    """
    data = fetch_olivetti_faces(data_home="./data", shuffle=True, random_state=0)
    images = data.images
    targets = data.target

    n_samples = images.shape[0]
    n_train = int(0.7 * n_samples)

    train_imgs, test_imgs = images[:n_train], images[n_train:]
    train_lbls, test_lbls = targets[:n_train], targets[n_train:]

    train_imgs = torch.from_numpy(train_imgs).unsqueeze(1).float()
    test_imgs  = torch.from_numpy(test_imgs).unsqueeze(1).float()
    train_lbls = torch.from_numpy(train_lbls).long()
    test_lbls  = torch.from_numpy(test_lbls).long()

    train_set = TensorDataset(train_imgs, train_lbls)
    test_set  = TensorDataset(test_imgs, test_lbls)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=shuffle)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader