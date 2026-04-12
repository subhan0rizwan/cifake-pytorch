import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

train_data_path = "C:/Users/HAROON TRADERS\Documents/DS Stuff/cifake-pytorch/data-folder/train"
test_data_path = "C:/Users/HAROON TRADERS\Documents/DS Stuff/cifake-pytorch/data-folder/test"
BATCH_SIZE = 64
transformIMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ]
)

train_data_main = torchvision.datasets.ImageFolder(root = train_data_path, transform = transformIMG)

total_size = len(train_data_main)
val_size = int(0.15 * total_size)
train_size = total_size - val_size

train_data, val_data = random_split(train_data_main, [train_size, val_size])

train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_data_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)

test_data = torchvision.datasets.ImageFolder(root = test_data_path, transform = transformIMG)
test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)

def main():
    for i, (X_train, y_train) in enumerate(train_data_loader):
        print(f"Batch {i+1}")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        x = X_train

if __name__ == "__main__":
    main()

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

    self.pool = nn.MaxPool2d(2, 2)

    self.gap = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(64, 2)

  def forward(self, X):

      X = self.pool(F.relu(self.conv1(X)))  # 256 → 128
      X = self.pool(F.relu(self.conv2(X)))  # 128 → 64
      X = self.pool(F.relu(self.conv3(X)))  # 64 → 32

      X = self.gap(X)                      # → (B, 64, 1, 1)
      X = torch.flatten(X, 1)              # → (B, 64)

      X = self.fc(X)

      return X


model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

import time

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 3
total_images = len(train_data_loader.dataset)

losses = []
val_losses = []

start_time = time.time()

for epoch in range(num_epochs):
    model.train()

    processed_images = 0
    pbar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

    # 🔹 TRAINING LOOP
    for batch_idx, (images, labels) in enumerate(pbar):
        batch_size = images.size(0)
        processed_images += batch_size

        images, labels = images.to(device), labels.to(device)

        scores = model(images)
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pbar.set_postfix({
            "Train Loss": f"{loss.item():.4f}",
            "Processed": f"{processed_images}/{total_images}"
        })

    # 🔹 VALIDATION LOOP (AFTER EPOCH)
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_data_loader)
    val_losses.append(val_loss)

    print(f"\nEpoch {epoch+1} Validation Loss: {val_loss:.4f}")

    best_val_loss = 0.5

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Saved new best model")

# 🔹 TIME TRACKING
end_time = time.time()
total_time = end_time - start_time

print("\nTraining Finished")
print(f"Total Time: {total_time:.2f} seconds")
print(f"Total Time: {total_time/60:.2f} minutes")

# 🔹 PLOT BOTH LOSSES
plt.plot(losses, label="Train Loss")
plt.plot(
    [i * len(train_data_loader) for i in range(len(val_losses))],
    val_losses,
    label="Validation Loss",
    marker='o'
)

plt.xlabel("Batch number")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

