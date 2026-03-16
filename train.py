import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(10),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
])

# Load full training set to split
full_dataset = datasets.ImageFolder("chest_xray/train", transform=transform)
val_dataset = datasets.ImageFolder("chest_xray/train", transform=val_transform)

# Split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

train_set = torch.utils.data.Subset(full_dataset, train_indices.indices)
val_set = torch.utils.data.Subset(val_dataset, val_indices.indices)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

# Count class distribution for weighted loss
class_counts = [0, 0]
for _, label in train_set:
  class_counts[label] += 1

total = sum(class_counts)
class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32)
print(f"Class counts: {dict(zip(full_dataset.classes, class_counts))}")
print(f"Class weights: {class_weights.tolist()}")

# Model setup — unfreeze layer4 + fc
model = models.resnet50(weights="IMAGENET1K_V1")

for param in model.parameters():
  param.requires_grad = False

for param in model.layer4.parameters():
  param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(
  list(model.layer4.parameters()) + list(model.fc.parameters()),
  lr=0.001,
)

num_epochs = 10

for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0

  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  train_loss = running_loss / len(train_loader)
  print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")

  model.eval()
  all_preds = []
  all_labels_list = []

  with torch.no_grad():
    for images, labels in val_loader:
      images = images.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      all_preds.extend(predicted.cpu().tolist())
      all_labels_list.extend(labels.tolist())

  classes = full_dataset.classes
  normal_idx = classes.index("NORMAL")
  pneumonia_idx = classes.index("PNEUMONIA")

  tp = sum(1 for p, l in zip(all_preds, all_labels_list) if p == pneumonia_idx and l == pneumonia_idx)
  tn = sum(1 for p, l in zip(all_preds, all_labels_list) if p == normal_idx and l == normal_idx)
  fp = sum(1 for p, l in zip(all_preds, all_labels_list) if p == pneumonia_idx and l == normal_idx)
  fn = sum(1 for p, l in zip(all_preds, all_labels_list) if p == normal_idx and l == pneumonia_idx)

  sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
  specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

  print(f"           Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}")

# Find optimal threshold on validation set
print("\nFinding optimal threshold on validation set...")
model.eval()
all_probs = []
all_labels_list = []

with torch.no_grad():
  for images, labels in val_loader:
    images = images.to(device)
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    all_probs.extend(probs[:, pneumonia_idx].cpu().tolist())
    all_labels_list.extend(labels.tolist())

best_threshold = 0.5
best_j = 0

for t in [i / 100 for i in range(101)]:
  preds = [pneumonia_idx if p >= t else normal_idx for p in all_probs]
  tp = sum(1 for p, l in zip(preds, all_labels_list) if p == pneumonia_idx and l == pneumonia_idx)
  tn = sum(1 for p, l in zip(preds, all_labels_list) if p == normal_idx and l == normal_idx)
  fp = sum(1 for p, l in zip(preds, all_labels_list) if p == pneumonia_idx and l == normal_idx)
  fn = sum(1 for p, l in zip(preds, all_labels_list) if p == normal_idx and l == pneumonia_idx)

  sens = tp / (tp + fn) if (tp + fn) > 0 else 0
  spec = tn / (tn + fp) if (tn + fp) > 0 else 0
  j = sens + spec - 1

  if j > best_j:
    best_j = j
    best_threshold = t

print(f"Optimal threshold (Youden's J): {best_threshold}")

torch.save({
  "model": model.state_dict(),
  "threshold": best_threshold,
  "classes": full_dataset.classes,
}, "model.pth")

print(f"Model and threshold saved. Classes: {full_dataset.classes}")
