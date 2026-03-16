import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import random

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

# Load full training set
full_dataset = datasets.ImageFolder("chest_xray/train", transform=transform)
val_dataset = datasets.ImageFolder("chest_xray/train", transform=val_transform)

# Separate indices by class
normal_indices = []
pneumonia_indices = []
for i, (_, label) in enumerate(full_dataset):
  if label == full_dataset.classes.index("NORMAL"):
    normal_indices.append(i)
  else:
    pneumonia_indices.append(i)

print(f"Original counts: NORMAL={len(normal_indices)}, PNEUMONIA={len(pneumonia_indices)}")

# Undersample pneumonia to match normal count
random.seed(42)
pneumonia_indices = random.sample(pneumonia_indices, len(normal_indices))
balanced_indices = normal_indices + pneumonia_indices
random.shuffle(balanced_indices)

print(f"After undersampling: {len(balanced_indices)} total ({len(normal_indices)} per class)")

# Split 60/20/20
train_end = int(0.6 * len(balanced_indices))
val_end = int(0.8 * len(balanced_indices))
train_indices = balanced_indices[:train_end]
val_indices = balanced_indices[train_end:val_end]
test_indices = balanced_indices[val_end:]

print(f"Split: {len(train_indices)} train / {len(val_indices)} val / {len(test_indices)} test")

train_set = Subset(full_dataset, train_indices)
val_set = Subset(val_dataset, val_indices)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

# Model setup — unfreeze layer4 + fc
model = models.resnet50(weights="IMAGENET1K_V1")

for param in model.parameters():
  param.requires_grad = False

for param in model.layer4.parameters():
  param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
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

  tp = 0
  tn = 0
  fp = 0
  fn = 0
  for p, l in zip(all_preds, all_labels_list):
    if p == pneumonia_idx and l == pneumonia_idx:
      tp += 1
    elif p == normal_idx and l == normal_idx:
      tn += 1
    elif p == pneumonia_idx and l == normal_idx:
      fp += 1
    elif p == normal_idx and l == pneumonia_idx:
      fn += 1

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

MIN_SENSITIVITY = 0.99
best_threshold = 0.01
best_spec = 0

thresholds = []
for i in range(101):
  thresholds.append(i / 100)

for t in thresholds:
  preds = []
  for p in all_probs:
    if p >= t:
      preds.append(pneumonia_idx)
    else:
      preds.append(normal_idx)

  tp = 0
  tn = 0
  fp = 0
  fn = 0
  for p, l in zip(preds, all_labels_list):
    if p == pneumonia_idx and l == pneumonia_idx:
      tp += 1
    elif p == normal_idx and l == normal_idx:
      tn += 1
    elif p == pneumonia_idx and l == normal_idx:
      fp += 1
    elif p == normal_idx and l == pneumonia_idx:
      fn += 1

  sens = tp / (tp + fn) if (tp + fn) > 0 else 0
  spec = tn / (tn + fp) if (tn + fp) > 0 else 0

  if sens >= MIN_SENSITIVITY and spec > best_spec:
    best_spec = spec
    best_threshold = t

print(f"Optimal threshold (sensitivity >= {MIN_SENSITIVITY}): {best_threshold}")

torch.save({
  "model": model.state_dict(),
  "threshold": best_threshold,
  "classes": full_dataset.classes,
  "test_indices": test_indices,
}, "model.pth")

print(f"Model and threshold saved. Classes: {full_dataset.classes}")
