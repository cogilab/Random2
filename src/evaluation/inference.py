import numpy as np
import torch
import torch.nn.functional as F

def test(model, loader, device = None):
    """
    Evaluate the model on the given data loader.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing the data to test on.
        device (torch.device): Device to use for testing.

    Returns:
        tuple: A tuple containing:
            - pred (np.ndarray): Boolean array indicating if each prediction is correct.
            - conf (np.ndarray): Confidence scores for each prediction.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    pred = []  # List to store prediction results
    conf = []  # List to store confidence scores

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Perform predictions using the model
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            pred.append((predicted == labels).cpu().numpy())
            conf.append(F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy())

    pred = np.concatenate(pred)
    conf = np.concatenate(conf)

    return pred, conf
