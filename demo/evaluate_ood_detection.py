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

# Helper functions for OOD evaluation

def cdf(data, bins=20, range=(0, 1)):
    hist, bin_edges = np.histogram(data, bins=bins, range=range, density=False)
    cdf_vals = np.cumsum(hist) / np.sum(hist)
    return cdf_vals, bin_edges

def roc_curve(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    scores = np.concatenate([a, b])
    labels = np.concatenate([np.ones_like(a, dtype=int), np.zeros_like(b, dtype=int)])

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)

    P = np.sum(labels)
    N = len(labels) - P
    distinct = np.r_[True, scores_sorted[1:] != scores_sorted[:-1]]
    idx = np.where(distinct)[0]

    fpr = np.r_[0.0, fps[idx] / (N if N > 0 else 1)]
    tpr = np.r_[0.0, tps[idx] / (P if P > 0 else 1)]

    return fpr, tpr

def auc(fpr, tpr):
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    assert fpr.shape == tpr.shape, "Length of input arrays must be the same."

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    return np.trapz(tpr, fpr)


# Baseline OOD detection performance

roc_control = roc_curve(conf_id_control, conf_ood_control)
roc_pretrain = roc_curve(conf_id_pretrain, conf_ood_pretrain)

baseline_auc_control = auc(roc_control[0], roc_control[1])
baseline_auc_pretrain = auc(roc_pretrain[0], roc_pretrain[1])

print("================")
print(f"Baseline OOD detection AUROC (w/o): {baseline_auc_control:.4f}")
print(f"Baseline OOD detection AUROC (w/) : {baseline_auc_pretrain:.4f}")

def energy_score(logits):
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    return np.log(np.sum(np.exp(logits), axis=1))

def compute_energy_scores_for_loader(loader, model, device):
    model.eval()
    logits = []
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            logit = model(data)
            logits.append(logit.cpu().numpy())
    
    logits = np.concatenate(logits, axis=0)
    scores = energy_score(logits)
    return scores                                                     

@torch.no_grad()
def temperature_scaling(model, x, T: float):
    model.eval()
    logits = model(x)
    probs = F.softmax(logits / T, dim=1)
    msp_scores, _ = torch.max(probs, dim=1)
    return msp_scores

def odin(
    model, x, T: float, epsilon: float,
    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
    energy=False
):
    model.eval()

    def denorm(z):
        m = torch.as_tensor(mean, device=z.device)[None, :, None, None]
        s = torch.as_tensor(std,  device=z.device)[None, :, None, None]
        return z * s + m

    def norm(z):
        m = torch.as_tensor(mean, device=z.device)[None, :, None, None]
        s = torch.as_tensor(std,  device=z.device)[None, :, None, None]
        return (z - m) / s

    x = x.clone().detach().requires_grad_(True)

    need = []
    for p in model.parameters():
        if p.requires_grad:
            need.append(p); p.requires_grad_(False)

    try:
        logits = model(x)
        scaled = logits / T
        _, yhat = torch.max(scaled, dim=1)
        loss = F.cross_entropy(scaled, yhat)
        loss.backward()

        s = torch.as_tensor(std, device=x.device)[None, :, None, None]
        grad_sign_pix = (x.grad.detach() / s).sign()

        x_pix = denorm(x.detach())
        x_pix_pert = torch.clamp(x_pix - epsilon * grad_sign_pix, 0.0, 1.0)

        with torch.no_grad():
            x_pert = norm(x_pix_pert)
            logits_perturbed = model(x_pert)
            
            if energy:
                scores = energy_score(logits_perturbed)
            else:
                probs = F.softmax(logits_perturbed / T, dim=1)
                scores, _ = probs.max(dim=1)
        return scores
    finally:
        for p in need: p.requires_grad_(True)
        model.zero_grad(set_to_none=True)

def _auroc_from_id_ood(id_scores, ood_scores):
    roc = roc_curve(id_scores, ood_scores)
    return auc(roc[0], roc[1])

def find_best_T(model, id_loader, ood_loader, T_list):
    best_t, best_auroc = None, -1.0
    device = next(model.parameters()).device

    for T in T_list:
        id_scores_all, ood_scores_all = [], []
        for (id_batch, *_), (ood_batch, *_) in zip(id_loader, ood_loader):
            id_batch = id_batch.to(device)
            ood_batch = ood_batch.to(device)
            with torch.no_grad():
                id_scores = temperature_scaling(model, id_batch, T)
                ood_scores = temperature_scaling(model, ood_batch, T)
            id_scores_all.extend(id_scores.detach().cpu().numpy())
            ood_scores_all.extend(ood_scores.detach().cpu().numpy())

        if id_scores_all and ood_scores_all:
            auroc = _auroc_from_id_ood(id_scores_all, ood_scores_all)
            #print(f"Testing T = {T}, AUROC = {auroc:.4f}")
            if auroc > best_auroc:
                best_auroc, best_t = auroc, T
        else:
            pass
            #print(f"Testing T = {T}, AUROC = N/A (no data)")

    #print(f"-> Best T: {best_t}, Best AUROC: {best_auroc:.4f}")
    return best_t

def find_best_odin_hyperparams(model, id_loader, ood_loader, T_list, epsilon_list, energy=False):
    best_t, best_eps, best_auroc = None, None, -1.0
    device = next(model.parameters()).device

    for T in T_list:
        for eps in epsilon_list:
            id_scores_all, ood_scores_all = [], []
            for (id_batch, *_), (ood_batch, *_) in zip(id_loader, ood_loader):
                id_batch = id_batch.to(device)
                ood_batch = ood_batch.to(device)

                id_scores = odin(model, id_batch, T, eps, energy=energy)
                ood_scores = odin(model, ood_batch, T, eps, energy=energy)

                if energy:
                    id_scores_all.extend(id_scores)
                    ood_scores_all.extend(ood_scores)
                else:
                    id_scores_all.extend(id_scores.detach().cpu().numpy())
                    ood_scores_all.extend(ood_scores.detach().cpu().numpy())

            if id_scores_all and ood_scores_all:
                auroc = _auroc_from_id_ood(id_scores_all, ood_scores_all)
                #print(f"Testing T = {T}, Epsilon = {eps}, AUROC = {auroc:.4f}")
                if auroc > best_auroc:
                    best_auroc, best_t, best_eps = auroc, T, eps
            else:
                #print(f"Testing T = {T}, Epsilon = {eps}, AUROC = N/A (no data)")
                pass

    #print(f"-> Best T: {best_t}, Best Epsilon: {best_eps}, Best AUROC: {best_auroc:.4f}")
    return best_t, best_eps

T_CANDIDATES = [1.5, 2, 5, 10, 20]
EPSILON_CANDIDATES = [0.001, 0.01, 0.02, 0.05]

T_control = find_best_T(model_control, cifar10_ood_val_loader, svhn_ood_val_loader, T_CANDIDATES)
T_odin_control, epsilon_odin_control = find_best_odin_hyperparams(model_control, cifar10_ood_val_loader, svhn_ood_val_loader, T_CANDIDATES, EPSILON_CANDIDATES)
T_pretrain = find_best_T(model_pretrain, cifar10_ood_val_loader, svhn_ood_val_loader, T_CANDIDATES)
T_odin_pretrain, epsilon_odin_pretrain = find_best_odin_hyperparams(model_pretrain, cifar10_ood_val_loader, svhn_ood_val_loader, T_CANDIDATES, EPSILON_CANDIDATES)

def temperature_scaling_scores(model, loader, T, device):
    """
    Computes the temperature-scaled scores for a DataLoader.

    Args:
        model (torch.nn.Module): The model used to extract logits.
        loader (torch.utils.data.DataLoader): The DataLoader containing the dataset.
        T (float): The temperature for scaling.
        device (torch.device): The device to run the computations on.

    Returns:
        np.ndarray: The temperature-scaled scores for each point in the dataset.
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            logits = model(data)
            scaled_scores = F.softmax(logits / T, dim=1).max(dim=1)[0]
            scores.extend(scaled_scores.cpu().numpy())
    
    return np.array(scores)

# Temperature scaling OOD detection performance

id_scores_control = temperature_scaling_scores(model_control, cifar10_ood_test_loader, T_control, device)
ood_scores_control = temperature_scaling_scores(model_control, svhn_ood_test_loader, T_control, device)
id_scores_pretrain = temperature_scaling_scores(model_pretrain, cifar10_ood_test_loader, T_pretrain, device)
ood_scores_pretrain = temperature_scaling_scores(model_pretrain, svhn_ood_test_loader, T_pretrain, device)

roc_temp_control = roc_curve(id_scores_control, ood_scores_control)
roc_temp_pretrain = roc_curve(id_scores_pretrain, ood_scores_pretrain)
temp_auc_control = auc(roc_temp_control[0], roc_temp_control[1])
temp_auc_pretrain = auc(roc_temp_pretrain[0], roc_temp_pretrain[1])

print("================")
print(f"Temperature scaling OOD detection AUROC (w/o): {temp_auc_control:.4f}")
print(f"Temperature scaling OOD detection AUROC (w/) : {temp_auc_pretrain:.4f}")

def odin_scores(model, loader, T, epsilon, device, energy=False):
    """
    Computes the ODIN scores for a DataLoader.

    Args:
        model (torch.nn.Module): The model used to extract logits.
        loader (torch.utils.data.DataLoader): The DataLoader containing the dataset.
        T (float): The temperature for scaling.
        epsilon (float): The perturbation size.
        device (torch.device): The device to run the computations on.

    Returns:
        np.ndarray: The ODIN scores for each point in the dataset.
    """
    model.eval()
    scores = []
    
    for data, _ in loader:
        data = data.to(device)
        output_score = odin(model, data, T, epsilon, energy=energy)
        
        if isinstance(output_score, torch.Tensor):
            scores.extend(output_score.cpu().numpy())
        else:
            scores.extend(output_score)
    
    return np.array(scores)

# ODIN OOD detection performance

id_scores_odin_control = odin_scores(model_control, cifar10_ood_test_loader, T_odin_control, epsilon_odin_control, device)
ood_scores_odin_control = odin_scores(model_control, svhn_ood_test_loader, T_odin_control, epsilon_odin_control, device)
id_scores_odin_pretrain = odin_scores(model_pretrain, cifar10_ood_test_loader, T_odin_pretrain, epsilon_odin_pretrain, device)
ood_scores_odin_pretrain = odin_scores(model_pretrain, svhn_ood_test_loader, T_odin_pretrain, epsilon_odin_pretrain, device)

roc_odin_control = roc_curve(id_scores_odin_control, ood_scores_odin_control)
roc_odin_pretrain = roc_curve(id_scores_odin_pretrain, ood_scores_odin_pretrain)

odin_auc_control = auc(roc_odin_control[0], roc_odin_control[1])
odin_auc_pretrain = auc(roc_odin_pretrain[0], roc_odin_pretrain[1])

print("================")
print(f"ODIN OOD detection AUROC (w/o): {odin_auc_control:.4f}")
print(f"ODIN OOD detection AUROC (w/) : {odin_auc_pretrain:.4f}")

# Energy-score OOD detection performance

id_scores_energy_control = odin_scores(model_control, cifar10_ood_test_loader, T_odin_control, epsilon_odin_control, device, energy=True)
ood_scores_energy_control = odin_scores(model_control, svhn_ood_test_loader, T_odin_control, epsilon_odin_control, device, energy=True)
id_scores_energy_pretrain = odin_scores(model_pretrain, cifar10_ood_test_loader, T_odin_pretrain, epsilon_odin_pretrain, device, energy=True)
ood_scores_energy_pretrain = odin_scores(model_pretrain, svhn_ood_test_loader, T_odin_pretrain, epsilon_odin_pretrain, device, energy=True)

roc_energy_odin_control = roc_curve(id_scores_energy_control, ood_scores_energy_control)
roc_energy_odin_pretrain = roc_curve(id_scores_energy_pretrain, ood_scores_energy_pretrain)
energy_auc_control = auc(roc_energy_odin_control[0], roc_energy_odin_control[1])
energy_auc_pretrain = auc(roc_energy_odin_pretrain[0], roc_energy_odin_pretrain[1])

print("================")
print(f"Energy-score OOD detection AUROC (w/o): {energy_auc_control:.4f}")
print(f"Energy-score OOD detection AUROC (w/) : {energy_auc_pretrain:.4f}")