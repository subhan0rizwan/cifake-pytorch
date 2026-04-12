from PIL import Image
import torch
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F

path = input("Enter image path: ").strip().replace('"', '')
img = Image.open(path).convert("RGB")

transformIMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
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

  def forward(self, X):

      X = self.pool(F.relu(self.conv1(X)))
      X = self.pool(F.relu(self.conv2(X)))
      X = self.pool(F.relu(self.conv3(X)))

      X = self.gap(X)                     
      X = torch.flatten(X, 1)
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