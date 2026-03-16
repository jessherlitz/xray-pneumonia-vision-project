import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Load checkpoint
checkpoint = torch.load("model.pth", weights_only=False)
THRESHOLD = checkpoint["threshold"]
test_indices = checkpoint["test_indices"]
classes = checkpoint["classes"]
print(f"Using threshold from validation: {THRESHOLD}")

# Load the same dataset and extract the held-out test split
full_dataset = datasets.ImageFolder("chest_xray/train", transform=transform)
test_set = Subset(full_dataset, test_indices)
test_loader = DataLoader(test_set, batch_size=16)

print(f"Test set size: {len(test_set)}")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model"])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.eval()

normal_idx = classes.index("NORMAL")
pneumonia_idx = classes.index("PNEUMONIA")

all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs[:, pneumonia_idx].cpu().tolist())
        all_labels.extend(labels.tolist())

# Apply threshold
all_predictions = []
for p in all_probs:
    if p >= THRESHOLD:
        all_predictions.append(pneumonia_idx)
    else:
        all_predictions.append(normal_idx)

tp = 0
tn = 0
fp = 0
fn = 0
for p, l in zip(all_predictions, all_labels):
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
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\nThreshold: {THRESHOLD}")
print(f"\nConfusion Matrix:")
print(f"                Predicted NORMAL  Predicted PNEUMONIA")
print(f"Actual NORMAL   {tn:>15}  {fp:>19}")
print(f"Actual PNEUMONIA{fn:>15}  {tp:>19}")
print(f"\nSensitivity (recall): {sensitivity:.3f}  — of all pneumonia cases, how many did we catch")
print(f"Specificity:          {specificity:.3f}  — of all normal cases, how many did we correctly ID")
print(f"Precision:            {precision:.3f}  — of all pneumonia predictions, how many were right")
print(f"\nTotal test images: {len(all_labels)}")

# ROC Curve
thresholds = []
for i in range(101):
    thresholds.append(i / 100)

sensitivities = []
one_minus_specificities = []

for t in thresholds:
    preds = []
    for p in all_probs:
        if p >= t:
            preds.append(pneumonia_idx)
        else:
            preds.append(normal_idx)

    tp_t = 0
    tn_t = 0
    fp_t = 0
    fn_t = 0
    for p, l in zip(preds, all_labels):
        if p == pneumonia_idx and l == pneumonia_idx:
            tp_t += 1
        elif p == normal_idx and l == normal_idx:
            tn_t += 1
        elif p == pneumonia_idx and l == normal_idx:
            fp_t += 1
        elif p == normal_idx and l == pneumonia_idx:
            fn_t += 1

    sens = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0

    sensitivities.append(sens)
    one_minus_specificities.append(1 - spec)

# AUC (trapezoidal rule)
auc = 0
for i in range(1, len(thresholds)):
    auc += (one_minus_specificities[i] - one_minus_specificities[i - 1]) * (sensitivities[i] + sensitivities[i - 1]) / 2
auc = abs(auc)

print(f"\nAUC: {auc:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(one_minus_specificities, sensitivities, color="blue", label=f"ROC Curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random (AUC = 0.5)")
plt.scatter([1 - specificity], [sensitivity], color="red", zorder=5, label=f"Optimal threshold ({THRESHOLD})")
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.title("ROC Curve — Pneumonia Detection")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
print("ROC curve saved to roc_curve.png")
