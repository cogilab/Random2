import torch
import numpy as np
from typing import Optional, Tuple

def random_input(input_shape, batch_size, mean=0, std=1, mode="normal"):
    """Generates random input data for training or validation."""
    if mode == "normal":
        inputs = torch.randn((batch_size, *input_shape)) * std + mean
    elif mode == "uniform":
        k = np.sqrt(12) * std
        q = mean - k / 2
        inputs = torch.rand((batch_size, *input_shape)) * k + q
    else:
        raise ValueError("Unknown mode: choose 'normal' or 'uniform'")

    return inputs

def random_label(batch_size, output_size, uniform_label=False):
    """Generates random labels for training or validation."""
    if uniform_label:
        labels = torch.full((batch_size,), 1/output_size)
    else:
        labels = torch.randint(0, output_size, (batch_size,))

    return labels


def random_train(model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 criterion: torch.nn.Module, 
                 input_shape: Tuple[int, ...], 
                 num_image: int, 
                 batch_size: int, 
                 output_size: int, 
                 mean: float = 0, 
                 std: float = 1, 
                 mode: str = "normal", 
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                 uniform_label: bool = False, 
                 device: Optional[torch.device] = None) -> Tuple[float, float]:
    """
    Trains the model on randomly generated data.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model's parameters.
        criterion (torch.nn.Module): Loss function to minimize.
        input_shape (Tuple): Shape of the input data excluding the batch size.
        num_image (int): Total number of images to simulate during training.
        batch_size (int): Number of samples per batch.
        output_size (int): Number of output classes.
        mean (float, optional): Mean for generating normal data. Default is 0.
        std (float, optional): Standard deviation for generating normal data. Default is 1.
        mode (str, optional): Distribution mode ("normal" or "uniform"). Default is "normal".
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler], optional): Learning rate scheduler. Default is None.
        uniform_label (bool, optional): If True, labels are uniformly distributed instead of one-hot encoding. Default is False.
        device (Optional[torch.device], optional): Device to perform computations. Default is inferred (cuda if available).

    Returns:
        Tuple[float, float]: Average loss and accuracy over the entire training dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    running_loss = []
    running_acc = []

    for _ in range(num_image // batch_size):
        inputs = random_input(input_shape, batch_size, mean, std, mode)
        labels = random_label(batch_size, output_size, uniform_label)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs.data, 1)
        running_loss.append(loss.item())
        running_acc.append((torch.sum(preds == labels.data).item()) / len(preds))

    epoch_loss = np.mean(running_loss)
    epoch_acc = np.mean(running_acc)
    
    if scheduler is not None:
        scheduler.step()

    return epoch_loss, epoch_acc

def random_validation(model: torch.nn.Module, 
                      criterion: torch.nn.Module, 
                      input_shape: Tuple[int, ...], 
                      num_image: int, 
                      batch_size: int, 
                      output_size: int, 
                      mean: float = 0, 
                      std: float = 1, 
                      mode: str = "normal", 
                      device: Optional[torch.device] = None) -> Tuple[float, float]:
    """
    Validates the model on randomly generated data.

    Args:
        model (torch.nn.Module): The model to validate.
        criterion (torch.nn.Module): Loss function for evaluation.
        input_shape (Tuple): Shape of the input data excluding the batch size.
        num_image (int): Total number of images to simulate during validation.
        batch_size (int): Number of samples per batch.
        output_size (int): Number of output classes.
        mean (float, optional): Mean for generating normal data. Default is 0.
        std (float, optional): Standard deviation for generating normal data. Default is 1.
        mode (str, optional): Distribution mode ("normal" or "uniform"). Default is "normal".
        device (Optional[torch.device], optional): Device to perform computations. Default is inferred (cuda if available).

    Returns:
        Tuple[float, float]: Average loss and accuracy over the entire validation dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    running_loss = []
    running_acc = []

    with torch.no_grad():
        for _ in range(num_image // batch_size):
            inputs = random_input(input_shape, batch_size, mean, std, mode)
            labels = random_label(batch_size, output_size)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)
            running_loss.append(loss.item())
            running_acc.append((torch.sum(preds == labels.data).item()) / len(preds))

    epoch_loss = np.mean(running_loss)
    epoch_acc = np.mean(running_acc)

    return epoch_loss, epoch_acc
