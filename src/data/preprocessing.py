import torch
from torchvision import transforms

def measure_mean_and_std(dataset):
    """Measure mean and standard deviation of a dataset."""
    mean = 0.
    std = 0.
    num_samples = 0.
    for data, _ in dataset:
        num_samples += 1
        mean += torch.mean(data)
        std += torch.std(data)
    mean /= num_samples
    std /= num_samples
    return mean, std

def get_transform(mean, std):
    """Get the transform to apply to the data."""
    
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
