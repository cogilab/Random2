import sys
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

datasets_path = '/dataset'
num_samples = 5000 # Number of samples per class
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load CIFAR-10 dataset

cifar10_mean = [0.49139968, 0.48215841, 0.44653091]
cifar10_std = [0.24703223, 0.24348513, 0.26158784]

cifar10_train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(cifar10_mean, cifar10_std)
])

cifar10_test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(cifar10_mean, cifar10_std)
])

cifar10_train = torchvision.datasets.CIFAR10(
    root=datasets_path, train=True, download=True,
    transform=cifar10_train_transform)
cifar10_test = torchvision.datasets.CIFAR10(
    root=datasets_path, train=False, download=True,
    transform=cifar10_test_transform)

cifar10_train_loader = torch.utils.data.DataLoader(
    cifar10_train, batch_size=128, shuffle=True,
    num_workers=0)
cifar10_test_loader = torch.utils.data.DataLoader(
    cifar10_test, batch_size=128, shuffle=False,
    num_workers=0)

def small_class_dataset(dataset, num_classes, num_samples):
    """
    Create a small dataset with a specified number of samples per class.
    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        num_classes (int): The number of classes in the dataset.
        num_samples (int): The number of samples to select from each class.
    Returns:
        torch.utils.data.Subset: A subset of the dataset containing the specified number of samples per class.
    """
    indices = []
    for i in range(num_classes):
        class_indices = np.where(np.array(dataset.targets) == i)[0]
        selected_indices = np.random.choice(class_indices, num_samples, replace=False)
        indices.extend(selected_indices)
    np.random.shuffle(indices)
    
    temp_data = dataset.data[indices]
    temp_targets = np.array(dataset.targets)[indices]

    small_dataset = copy.deepcopy(dataset)
    small_dataset.data = temp_data
    small_dataset.targets = temp_targets.tolist()

    return small_dataset

cifar10_train_small = small_class_dataset(cifar10_train, num_classes=10, num_samples=num_samples)
cifar10_train_loader = torch.utils.data.DataLoader(
    cifar10_train_small, batch_size=256, shuffle=True,
    num_workers=0)

from src.training.random_training import random_train, random_validation
from src.training.training import train, validation
from src.training.random_training import random_noise_dataset
from src.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pretrain = resnet18(num_classes=10).to(device)
model_control = resnet18(num_classes=10).to(device)

lr = 0.01
momentum = 0.9
weight_decay = 0.01

optimizer_pretrain = torch.optim.SGD(model_pretrain.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer_control = torch.optim.SGD(model_control.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

loss_fn = nn.CrossEntropyLoss()

training_info = {"train_loss": [], "train_acc": [],
                 "test_loss": [], "test_acc": [],
                 "train_conf": [], "test_conf": [],
                 "train_ece": [], "test_ece": [],
                 "best_model": None}

info_pretrain = copy.deepcopy(training_info)
info_control = copy.deepcopy(training_info)


# random noise warm-up training

epochs_pretrain = 5

input_shape = (3, 32, 32)
num_image = 50000
BATCH_SIZE = 256
output_size = 10

for epoch in range(epochs_pretrain):    
    if epoch == 0:
        train_loss, train_acc = random_validation(model_pretrain, loss_fn,
                                                  input_shape, num_image, BATCH_SIZE, output_size, mean=0, std=1)
    else:
        train_loss, train_acc = random_train(model_pretrain, optimizer_pretrain, loss_fn, 
                                            input_shape, num_image, BATCH_SIZE, output_size, mean=0, std=1)
        
    info_pretrain["train_loss"].append(train_loss)
    info_pretrain["train_acc"].append(train_acc)

    print(f"Epoch: {epoch + 1}/{epochs_pretrain}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

# downstream training

lr = 0.1
momentum = 0.9
weight_decay = 1e-4

epochs_train = 50

optimizer_pretrain = torch.optim.SGD(model_pretrain.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer_control = torch.optim.SGD(model_control.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

scheduler_pretrain = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pretrain, T_max=epochs_train)
scheduler_control = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_control, T_max=epochs_train)

# conventional model training
best_loss = float('inf')

for epoch in range(epochs_train):
    if epoch == 0:
        train_loss, train_acc = validation(model_control, cifar10_train_loader, loss_fn)
    else:
        train_loss, train_acc = train(model_control, cifar10_train_loader, optimizer_control, loss_fn, scheduler_control)
    test_loss, test_acc = validation(model_control, cifar10_test_loader, loss_fn)
    info_control["train_loss"].append(train_loss)
    info_control["train_acc"].append(train_acc)
    info_control["test_loss"].append(test_loss)
    info_control["test_acc"].append(test_acc)
    print(f"Epoch: {epoch + 1}/{epochs_train}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
# warm-up + downstream training
best_loss = float('inf')

for epoch in range(epochs_train):
    if epoch == 0:
        train_loss, train_acc = validation(model_pretrain, cifar10_train_loader, loss_fn)
    else:
        train_loss, train_acc = train(model_pretrain, cifar10_train_loader, optimizer_pretrain, loss_fn, scheduler_pretrain)
    test_loss, test_acc = validation(model_pretrain, cifar10_test_loader, loss_fn)
    info_pretrain["train_loss"].append(train_loss)
    info_pretrain["train_acc"].append(train_acc)
    info_pretrain["test_loss"].append(test_loss)
    info_pretrain["test_acc"].append(test_acc)
    print(f"Epoch: {epoch + 1}/{epochs_train}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


torch.save(model_control.state_dict(), os.path.join('demo/pretrained/model_wo_random.pth'))
torch.save(model_pretrain.state_dict(), os.path.join('demo/pretrained/model_w_random.pth'))