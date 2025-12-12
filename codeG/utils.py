
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png') or img.endswith('.jpg')]
        # Preload all images into RAM as tensors
        self.images = []
        # Use multithreading to speed up image loading

        def load_image(img_path):
            with Image.open(img_path) as image:
                image = image.convert('RGB')  # or 'RGB'
                if self.transform:
                    image = self.transform(image)
            return image

        with ThreadPoolExecutor() as executor:
            self.images = list(tqdm(executor.map(load_image, self.image_paths), total=len(self.image_paths), desc="Loading images"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_dataloaders(args):
    """Create data loaders based on dataset configuration."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.n, args.n), antialias=True),
        transforms.Grayscale() if args.grayscale else transforms.Lambda(lambda x: x)
    ])
    
    if args.dataset == 'mnist':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'fashionmnist':
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'cifar10':
        root = './data'
        already = os.path.isdir(os.path.join(root, 'cifar-10-batches-py'))
        trainset = datasets.CIFAR10(root=root, train=True, download=not already, transform=transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = datasets.CIFAR10(root=root, train=False, download=not already, transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
        
    elif args.dataset == 'celeba':
        dataset = ImageDataset(r'C:\Roman\datasets\CelebA2\train', transform=transform)        
        trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataset_test = ImageDataset(r'C:\Roman\datasets\CelebA2\test', transform=transform)        
        testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'div2k':
        dataset = ImageDataset(r'C:\Roman\datasets\DIV2K_patches\patches\train', transform=transform)        
        trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataset_test = ImageDataset(r'C:\Roman\datasets\DIV2K_patches\patches\valid', transform=transform)        
        testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    return trainloader, testloader




def get_dataloaders_testing(args):
    """Create data loaders based on dataset configuration."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.n, args.n), antialias=True),
        transforms.Grayscale() if args.grayscale else transforms.Lambda(lambda x: x)
    ])
    
    if args.dataset == 'mnist':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'fashionmnist':
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'cifar10':
        root = './data'
        already = os.path.isdir(os.path.join(root, 'cifar-10-batches-py'))
        trainset = datasets.CIFAR10(root=root, train=True, download=not already, transform=transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testset = datasets.CIFAR10(root=root, train=False, download=not already, transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
        
    elif args.dataset == 'celeba':
       
        dataset_test = ImageDataset(r'C:\Roman\datasets\CelebA2\test', transform=transform)        
        testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'div2k':

        dataset_test = ImageDataset(r'C:\Roman\datasets\DIV2K_patches\patches\valid', transform=transform)        
        testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    return testloader





#### Utility functions for matrix generation

def generate_matrix(m,n):

    path = f"matrices/fixed_matrix_{m}_{n}.pt"
    if os.path.exists(path):
        fixed_matrix = torch.load(path)
    else:
        fixed_matrix = torch.round(torch.rand(m, n, device='cuda')) * 2 - 1
        torch.save(fixed_matrix, path)
    return fixed_matrix


def generate_orthogonal_rows_qr(A: torch.Tensor, mB: int, device='cuda') -> torch.Tensor:
    mA, n = A.shape
    path = f"matrices/orthogonal_rows_{mA}_{mB}_{n}.pt"
    if os.path.exists(path):
        return torch.load(path)
    else:
        if mB == 0:
            return torch.empty((0, n), device=device, dtype=A.dtype)
        if mB > (n - mA):
            print(f"No se pueden extraer {mB} vectores de un nullspace de dim {n-mA}")

        # QR completa sobre A^T para tener Q_full (n×n)
        Q_full, _       = torch.linalg.qr(A.T, mode='complete')  # (n, n)
        nullspace_basis = Q_full[:, mA:]                         # (n, n-mA)

        # combinaciones ortonormales aleatorias dentro del nullspace
        P  = torch.randn(nullspace_basis.shape[1], mB, device=device, dtype=A.dtype)
        U, _ = torch.linalg.qr(P)     # Q reducido: (n-mA, mB)
        # cada fila nueva
        new_rows = U.T.matmul(nullspace_basis.T)  # (mB, n)
        torch.save(new_rows, path)
        return new_rows