import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np

from medmnist import PathMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torchattacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def get_resnet(num_classes=9):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
test_dataset = PathMNIST(split='test', download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64)

# Inference utility
def get_preds(model, loader, attack=None):
    model.eval()
    y_true, y_pred = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        if attack:
            imgs = attack(imgs, labels)

        with torch.no_grad():
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    return np.array(y_true), np.array(y_pred)

# Confusion matrix plot
def plot_conf_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")

def main():
    os.makedirs("results", exist_ok=True)
    model = get_resnet(num_classes=9).to(device)
    model.load_state_dict(torch.load("blackbox_model.pth", map_location=device))

    # Clean
    y_true, y_pred = get_preds(model, test_loader)
    plot_conf_matrix(y_true, y_pred, "Confusion Matrix - Clean", "results/cm_clean.png")

    # FGSM
    fgsm = torchattacks.FGSM(model, eps=0.1)
    y_true, y_pred = get_preds(model, test_loader, attack=fgsm)
    plot_conf_matrix(y_true, y_pred, "Confusion Matrix - FGSM (ε=0.1)", "results/cm_fgsm.png")

    # PGD
    pgd = torchattacks.PGD(model, eps=0.1, alpha=2/255, steps=10)
    y_true, y_pred = get_preds(model, test_loader, attack=pgd)
    plot_conf_matrix(y_true, y_pred, "Confusion Matrix - PGD (ε=0.1)", "results/cm_pgd.png")

if __name__ == "__main__":
    main()
