import torch
import numpy as np

def train(model, train_loader, optimizer, criterion, scheduler=None, device=None):
    """
    Trains the model for one epoch on the provided data loader.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader providing training data.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        criterion (torch.nn.modules.loss._Loss): Loss function to minimize.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Default is None.
        device (torch.device, optional): Device to perform computations. Default is inferred (cuda if available).

    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    running_loss = []
    running_acc = []

    for i, (inputs, labels) in enumerate(train_loader):
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

def validation(model, test_loader, criterion, device=None):
    """
    Validates the model on the provided data loader.

    Args:
        model (torch.nn.Module): The model to validate.
        test_loader (torch.utils.data.DataLoader): DataLoader providing test data.
        criterion (torch.nn.modules.loss._Loss): Loss function to evaluate.

    Returns:
        tuple: Average loss and accuracy for the validation dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    running_loss = []
    running_acc = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)
            running_loss.append(loss.item())
            running_acc.append((torch.sum(preds == labels.data).item()) / len(preds))

    epoch_loss = np.mean(running_loss)
    epoch_acc = np.mean(running_acc)

    return epoch_loss, epoch_acc
