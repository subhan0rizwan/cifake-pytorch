from PIL import Image
import torch
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F

path = input("Enter image path: ").strip().replace('"', '')
img = Image.open(path).convert("RGB")

IMG_SIZE = 224

transformIMG = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale = (0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomAdjustSharpness(2, p=0.3),

    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ]
)

img_tensor = transformIMG(img)

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

    self.pool = nn.MaxPool2d(2, 2)
    self.gap = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(64, 2)
    self.dropout = nn.Dropout(0.5)

  def forward(self, X):

      X = self.pool(F.relu(self.conv1(X)))  # 256 → 128
      X = self.pool(F.relu(self.conv2(X)))  # 128 → 64
      X = self.pool(F.relu(self.conv3(X)))  # 64 → 32

      X = self.gap(X)                      # → (B, 64, 1, 1)
      X = torch.flatten(X, 1)              # → (B, 64)

      X = self.fc(X)

      return X

model = MyModel()

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

with torch.no_grad():
    output = model(img_tensor.unsqueeze(0))
    probs = torch.softmax(output, dim=1)

pred = torch.argmax(probs, dim=1)
confidence = probs[0][pred.item()].item()

classes = ["AI Generated", "REAL"]
print("Prediction:", classes[pred.item()])
print("Confidence:", round(confidence * 100, 2), "%")