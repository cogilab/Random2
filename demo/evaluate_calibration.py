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

def split_dataset(dataset, validation_size=5000, test_size=5000):
    """
    Split the dataset into validation and test sets.
    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        validation_size (int): The number of samples for the validation set.
        test_size (int): The number of samples for the test set.
    Returns:
        tuple: A tuple containing the validation and test datasets.
    """
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    val_indices = indices[:validation_size]
    test_indices = indices[validation_size:validation_size + test_size]
    
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return val_dataset, test_dataset


cifar10_train_small = small_class_dataset(cifar10_train, num_classes=10, num_samples=num_samples)
cifar10_train_loader = torch.utils.data.DataLoader(
    cifar10_train_small, batch_size=256, shuffle=True,
    num_workers=0)

svhn_test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(cifar10_mean, cifar10_std)
])

svhn_test = torchvision.datasets.SVHN(
    root=datasets_path, split='test', download=True,
    transform=svhn_test_transform)

svhn_test.data = svhn_test.data[:10000]
svhn_test.targets = svhn_test.labels[:10000]

svhn_test_loader = torch.utils.data.DataLoader(
    svhn_test, batch_size=128, shuffle=False,
    num_workers=0)

cifar10_ood_val, cifar10_ood_test = split_dataset(cifar10_test, validation_size=5000, test_size=5000)
svhn_ood_val, svhn_ood_test = split_dataset(svhn_test, validation_size=5000, test_size=5000)

cifar10_ood_test_loader = torch.utils.data.DataLoader(cifar10_ood_test, batch_size=128, shuffle=False, num_workers=0)
cifar10_ood_val_loader = torch.utils.data.DataLoader(cifar10_ood_val, batch_size=128, shuffle=False, num_workers=0)
svhn_ood_test_loader = torch.utils.data.DataLoader(svhn_ood_test, batch_size=128, shuffle=False, num_workers=0)
svhn_ood_val_loader = torch.utils.data.DataLoader(svhn_ood_val, batch_size=128, shuffle=False, num_workers=0)


from src.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_control = resnet18(num_classes=10)
model_control.load_state_dict(torch.load('demo/pretrained/model_wo_random.pth', map_location=device))
model_control.to(device)
model_control.eval()

model_pretrain = resnet18(num_classes=10)
model_pretrain.load_state_dict(torch.load('demo/pretrained/model_w_random.pth', map_location=device))
model_pretrain.to(device)
model_pretrain.eval()

from src.evaluation.calibration import reliability_diagram, ece
from src.evaluation.inference import inference as inference

pred_id_control, conf_id_control = inference(model_control, cifar10_ood_test_loader)
pred_id_pretrain, conf_id_pretrain = inference(model_pretrain, cifar10_ood_test_loader)

pred_ood_control, conf_ood_control = inference(model_control, svhn_ood_test_loader)
pred_ood_pretrain, conf_ood_pretrain = inference(model_pretrain, svhn_ood_test_loader)

acc, conf, num_sample = reliability_diagram(pred_id_control, conf_id_control, num_bin=10)
baseline_ece_control = ece(acc, conf, num_sample)

acc, conf, num_sample = reliability_diagram(pred_id_pretrain, conf_id_pretrain, num_bin=10)
baseline_ece_pretrain = ece(acc, conf, num_sample)

print("================")
print(f"Baseline ECE (w/o): {baseline_ece_control:.4f}")
print(f"Baseline ECE (w/) : {baseline_ece_pretrain:.4f}")

# Temperature scaling

def find_best_T_calibration(model, id_loader, T_list):
    best_t = None
    best_ece = float('inf')
    device = next(model.parameters()).device

    for T in T_list:
        id_scores_all = []
        id_preds_all = []
        
        with torch.no_grad():
            for data, labels in id_loader:
                data = data.to(device)
                
                logits = model(data)
                probs = F.softmax(logits / T, dim=1)

                confidences, predictions = torch.max(probs, dim=1)
                
                id_scores_all.extend(confidences.cpu().numpy())
                correct_preds = predictions.eq(labels.to(device))
                id_preds_all.extend(correct_preds.cpu().numpy())

        id_scores_all = np.array(id_scores_all)
        id_preds_all = np.array(id_preds_all)

        acc, conf, num_sample = reliability_diagram(id_preds_all, id_scores_all, num_bin=10)
        current_ece = ece(acc, conf, num_sample)
        
        # print(f"Testing T = {T}, ECE = {current_ece:.4f}")
        
        if current_ece < best_ece:
            best_ece = current_ece
            best_t = T
            
    # print(f"-> Best T: {best_t}, Best ECE: {best_ece:.4f}")
    return best_t

T_CALIBRATION_CANDIDATES = [1, 1.05, 1.1, 1.2, 1.5, 2, 3, 5]
T_calibration_control = find_best_T_calibration(model_control, cifar10_ood_val_loader, T_CALIBRATION_CANDIDATES)
T_calibration_pretrain = find_best_T_calibration(model_pretrain, cifar10_ood_val_loader, T_CALIBRATION_CANDIDATES)

def ece_temperature_scaling_calibration(model, loader, T, device):
    """
    Apply temperature scaling for calibration and compute ECE.
    
    Args:
        model (torch.nn.Module): The model to calibrate.
        loader (torch.utils.data.DataLoader): The DataLoader containing the dataset.
        T (float): The temperature for scaling.
        device (torch.device): The device to run the computations on.
    
    Returns:
        float: The ECE value after temperature scaling.
    """
    model.eval()
    all_scores = []
    all_preds = []
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits / T, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            all_scores.extend(confidences.cpu().numpy())
            correct_preds = predictions.eq(labels.to(device))
            all_preds.extend(correct_preds.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)

    acc, conf, num_sample = reliability_diagram(all_preds, all_scores, num_bin=10)
    ece_value = ece(acc, conf, num_sample)
    
    # print(f"ECE after temperature scaling: {ece_value:.4f}")
    return ece_value

# apply temperature scaling for calibration
temp_ece_control = ece_temperature_scaling_calibration(model_control, cifar10_ood_test_loader, T_calibration_control, device)
temp_ece_pretrain = ece_temperature_scaling_calibration(model_pretrain, cifar10_ood_test_loader, T_calibration_pretrain, device)

print("================")
print(f"Temperature scaling ECE (w/o): {temp_ece_control:.4f}")
print(f"Temperature scaling ECE (w/): {temp_ece_pretrain:.4f}")

# vector scaling
class VectorScaling(nn.Module):
    """
    W * logits + b
    """
    def __init__(self, num_classes):
        super(VectorScaling, self).__init__()
        self.W = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits):
        return logits * self.W + self.b
    
def train_vector_scaling(model, id_loader, num_classes=10, max_iter=50):
    device = next(model.parameters()).device
    model.eval()

    vs_model = VectorScaling(num_classes).to(device)
    
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for data, labels in id_loader:
            logits = model(data.to(device))
            all_logits.append(logits)
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    optimizer = torch.optim.LBFGS(vs_model.parameters(), lr=0.0005, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = vs_model(all_logits)
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    return vs_model

vs_model_control = train_vector_scaling(model_control, cifar10_ood_val_loader)
vs_model_pretrain = train_vector_scaling(model_pretrain, cifar10_ood_val_loader)

def evaluate_ece_with_scaling(model, scaling_model, loader):
    device = next(model.parameters()).device
    model.eval()
    scaling_model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in loader:
            logits = model(data.to(device))
            scaled_logits = scaling_model(logits)
            probs = F.softmax(scaled_logits, dim=1)
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    predictions = np.argmax(all_probs, axis=1)
    correct_preds = (predictions == all_labels)
    confidences = np.max(all_probs, axis=1)
    
    acc, conf, num_sample = reliability_diagram(correct_preds, confidences, num_bin=10)
    ece_value = ece(acc, conf, num_sample)
    
    return ece_value


vs_ece_control = evaluate_ece_with_scaling(model_control, vs_model_control, cifar10_ood_test_loader)
vs_ece_pretrain = evaluate_ece_with_scaling(model_pretrain, vs_model_pretrain, cifar10_ood_test_loader)

print("================")
print(f"Vector scaling ECE (w/o): {vs_ece_control:.4f}")
print(f"Vector scaling ECE (w/): {vs_ece_pretrain:.4f}")

# isotonic regression
import torch
import numpy as np
from sklearn.isotonic import IsotonicRegression


def train_isotonic_regression(model, id_loader):
    device = next(model.parameters()).device
    model.eval()

    all_confidences = []
    all_correctness = []
    
    with torch.no_grad():
        for data, labels in id_loader:
            logits = model(data.to(device))
            probs = torch.softmax(logits, dim=1)
            
            confidences, predictions = torch.max(probs, dim=1)
            correct_preds = predictions.eq(labels.to(device))
            
            all_confidences.append(confidences.cpu())
            all_correctness.append(correct_preds.cpu())

    all_confidences = torch.cat(all_confidences).numpy()
    all_correctness = torch.cat(all_correctness).numpy().astype(int)

    ir_model = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir_model.fit(all_confidences, all_correctness)
    
    return ir_model

ir_model_control = train_isotonic_regression(model_control, cifar10_ood_val_loader)
ir_model_pretrain = train_isotonic_regression(model_pretrain, cifar10_ood_val_loader)

def evaluate_ece_with_isotonic_regression(model, ir_model, loader):
    device = next(model.parameters()).device
    model.eval()
    
    original_confidences = []
    calibrated_confidences = []
    all_correctness = []
    
    with torch.no_grad():
        for data, labels in loader:
            logits = model(data.to(device))
            probs = torch.softmax(logits, dim=1)
            
            confidences, predictions = torch.max(probs, dim=1)
            correct_preds = predictions.eq(labels.to(device))
            
            original_confidences.append(confidences.cpu())
            all_correctness.append(correct_preds.cpu())
    
    original_confidences = torch.cat(original_confidences).numpy()
    all_correctness = torch.cat(all_correctness).numpy()
    
    calibrated_confidences = ir_model.transform(original_confidences)
    
    acc, conf, num_sample = reliability_diagram(all_correctness, calibrated_confidences, num_bin=15)
    ece_value = ece(acc, conf, num_sample)
    
    return ece_value

ir_ece_control = evaluate_ece_with_isotonic_regression(model_control, ir_model_control, cifar10_ood_test_loader)
ir_ece_pretrain = evaluate_ece_with_isotonic_regression(model_pretrain, ir_model_pretrain, cifar10_ood_test_loader)

print("================")
print(f"Isotonic regression ECE (w/o): {ir_ece_control:.4f}")
print(f"Isotonic regression ECE (w/): {ir_ece_pretrain:.4f}")