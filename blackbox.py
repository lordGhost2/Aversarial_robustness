import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from medmnist import PathMNIST
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torchattacks
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)


def load_dataset(split='train'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    dataset = PathMNIST(split=split, download=True, transform=transform)
    return dataset


def get_resnet(num_classes=9):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(model, loader, optimizer, epochs=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")


def save_report(y_true, y_pred, prefix):
    report = classification_report(y_true, y_pred, output_dict=True, target_names=[f"Class {i}" for i in range(9)])
    df = pd.DataFrame(report).transpose()
    df.to_excel(f"results/{prefix}_report.xlsx")
    df.to_csv(f"results/{prefix}_report.csv")
    print(f"✅ Saved: results/{prefix}_report.(xlsx/csv)")

    # Save confusion matrix (only for clean)
    if prefix == "clean":
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(9), yticklabels=range(9))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Clean)")
        plt.savefig("results/confusion_matrix.png", dpi=300)
        plt.close()
        print("✅ Saved: results/confusion_matrix.png")


def evaluate(model, loader, prefix="clean", attack=None):
    model.eval()
    y_true, y_pred = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
        if attack:
            imgs = attack(imgs, labels)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    acc = 100 * (sum([p == t for p, t in zip(y_pred, y_true)]) / len(y_true))
    print(f"\n📊 {prefix.upper()} Accuracy: {acc:.2f}%")
    print(classification_report(y_true, y_pred))
    save_report(y_true, y_pred, prefix)
    return acc


def main():
    train_dataset = load_dataset('train')
    test_dataset = load_dataset('test')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = get_resnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, optimizer, epochs=5)
    torch.save(model.state_dict(), "blackbox_model.pth")
    print("✅ Model saved to blackbox_model.pth")

    clean_acc = evaluate(model, test_loader, prefix="clean")

    fgsm = torchattacks.FGSM(model, eps=0.1)
    fgsm_acc = evaluate(model, test_loader, prefix="fgsm_eps0.1", attack=fgsm)

    pgd = torchattacks.PGD(model, eps=0.1, alpha=2/255, steps=10)
    pgd_acc = evaluate(model, test_loader, prefix="pgd_eps0.1", attack=pgd)

  
    print("\n📌 Summary:")
    print(f"Clean Accuracy:  {clean_acc:.2f}%")
    print(f"FGSM Accuracy:   {fgsm_acc:.2f}%")
    print(f"PGD Accuracy:    {pgd_acc:.2f}%")

if __name__ == "__main__":
    main()
