# -*- coding: utf-8 -*-
"""
Controle Qualite Industriel (Computer Vision)
Dataset : Casting Product (Kaggle) -- Pieces moulees OK vs Defaut
"""

import os
import random
import warnings
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    fbeta_score, recall_score, precision_score
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"[OK] Device : {DEVICE}")
print(f"[OK] PyTorch {torch.__version__}")

import kagglehub

print("Telechargement du dataset Casting Product depuis Kaggle...")
print("(Necessite un compte Kaggle. En Colab : Settings > Secrets > KAGGLE_USERNAME / KAGGLE_KEY)")

try:
    dataset_path = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")
    print(f"[OK] Dataset telecharge dans : {dataset_path}")
except Exception as e:
    print(f"[ERREUR] {e}")
    print("Telechargez manuellement depuis :")
    print("https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
    dataset_path = "./casting_data"

# Explorer la structure
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(str(dataset_path), "").count(os.sep)
    if level < 3:
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/  ({len(files)} fichiers)")

    """Localise les dossiers ok_front et def_front dans le dataset."""
    base = Path(base_path)
    result = {"ok": None, "defect": None}

    for dirpath in base.rglob("*"):
        if dirpath.is_dir():
            name = dirpath.name.lower()
            if "ok" in name and "front" in name:
                result["ok"] = str(dirpath)
            elif "def" in name and "front" in name:
                result["defect"] = str(dirpath)

    # Fallback : chercher dans train/test
    if result["ok"] is None:
        for dirpath in base.rglob("*"):
            if dirpath.is_dir():
                name = dirpath.name.lower()
                if name == "ok_front":
                    result["ok"] = str(dirpath)
                elif name == "def_front":
                    result["defect"] = str(dirpath)

    return result


dirs = find_image_dirs(dataset_path)
print(f"Dossier OK     : {dirs['ok']}")
print(f"Dossier Defect : {dirs['defect']}")

# Compter les images
ok_images = []
defect_images = []

if dirs["ok"]:
    ok_images = sorted([str(p) for p in Path(dirs["ok"]).glob("*.jpeg")]
                       + [str(p) for p in Path(dirs["ok"]).glob("*.jpg")]
                       + [str(p) for p in Path(dirs["ok"]).glob("*.png")])

if dirs["defect"]:
    defect_images = sorted([str(p) for p in Path(dirs["defect"]).glob("*.jpeg")]
                           + [str(p) for p in Path(dirs["defect"]).glob("*.jpg")]
                           + [str(p) for p in Path(dirs["defect"]).glob("*.png")])

print(f"\nImages OK     : {len(ok_images)}")
print(f"Images Defect : {len(defect_images)}")
print(f"Total         : {len(ok_images) + len(defect_images)}")
print(f"Ratio OK/Defect : {len(ok_images) / max(len(defect_images), 1):.2f}")

# Afficher des exemples
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for i in range(5):
    if i < len(ok_images):
        img = cv2.imread(ok_images[i], cv2.IMREAD_GRAYSCALE)
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"OK #{i+1}", fontsize=9)
    axes[0, i].axis("off")

for i in range(5):
    if i < len(defect_images):
        img = cv2.imread(defect_images[i], cv2.IMREAD_GRAYSCALE)
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(f"Defaut #{i+1}", fontsize=9)
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("OK", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Defaut", fontsize=12, fontweight="bold")
plt.suptitle("Exemples du dataset Casting Product", fontsize=14)
plt.tight_layout()
plt.show()


def preprocess_image(image: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
    """Pretraitement adapte aux images industrielles.
    
    CLAHE (Contrast Limited Adaptive Histogram Equalization) rehausse
    le contraste local, ce qui fait ressortir les defauts subtils.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    resized = cv2.resize(gray, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb


# Visualiser l'effet du CLAHE
if defect_images:
    sample = cv2.imread(defect_images[0], cv2.IMREAD_GRAYSCALE)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe_obj.apply(sample)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(sample, cmap="gray")
    ax1.set_title("Image originale")
    ax1.axis("off")
    ax2.imshow(enhanced, cmap="gray")
    ax2.set_title("Apres CLAHE (contraste rehausse)")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()

print("[OK] Preprocessing defini.")

    """Split les donnees en train/val en preservant la distribution des classes."""
    random.shuffle(ok_paths)
    random.shuffle(defect_paths)

    n_ok_val = int(len(ok_paths) * val_ratio)
    n_def_val = int(len(defect_paths) * val_ratio)

    val_paths = ok_paths[:n_ok_val] + defect_paths[:n_def_val]
    val_labels = [0] * n_ok_val + [1] * n_def_val

    train_paths = ok_paths[n_ok_val:] + defect_paths[n_def_val:]
    train_labels = [0] * (len(ok_paths) - n_ok_val) + [1] * (len(defect_paths) - n_def_val)

    return train_paths, train_labels, val_paths, val_labels


train_paths, train_labels, val_paths, val_labels = split_dataset(
    ok_images.copy(), defect_images.copy(), val_ratio=0.2
)

print(f"Train : {len(train_paths)} images (OK={train_labels.count(0)}, Defect={train_labels.count(1)})")
print(f"Val   : {len(val_paths)} images (OK={val_labels.count(0)}, Defect={val_labels.count(1)})")


class CastingDataset(Dataset):
    """Dataset PyTorch avec augmentations Albumentations."""

    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = preprocess_image(img)

        if self.augment:
            transformed = self.train_transform(image=img)
        else:
            transformed = self.val_transform(image=img)

        return transformed["image"], torch.tensor(self.labels[idx], dtype=torch.long)


BATCH_SIZE = 32

train_dataset = CastingDataset(train_paths, train_labels, augment=True)
val_dataset = CastingDataset(val_paths, val_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Verifier un batch
batch_imgs, batch_labels = next(iter(train_loader))
print(f"\nBatch shape : {batch_imgs.shape}")
print(f"Labels      : {batch_labels[:10].tolist()}")
print("[OK] DataLoaders prets.")

    """ResNet-18 avec tete de classification custom pour OK/Defaut."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def freeze_backbone(self):
        """Gele toutes les couches sauf la tete FC."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self, n_layers=2):
        """Degele les n dernieres couches convolutives."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        layers = [self.backbone.layer1, self.backbone.layer2,
                  self.backbone.layer3, self.backbone.layer4]
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Parametres entrainables : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, x):
        return self.backbone(x)


model = DefectClassifier(num_classes=2, pretrained=True).to(DEVICE)
print(f"[OK] Modele charge sur {DEVICE}")

# Poids de classe pour compenser le desequilibre
n_ok_train = train_labels.count(0)
n_def_train = train_labels.count(1)
weight_ok = len(train_labels) / (2 * max(n_ok_train, 1))
weight_def = len(train_labels) / (2 * max(n_def_train, 1))
class_weights = torch.tensor([weight_ok, weight_def], dtype=torch.float32).to(DEVICE)
print(f"  Poids de classe : OK={weight_ok:.3f}, Defect={weight_def:.3f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

    """Entraine le modele sur une epoch."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return epoch_loss, epoch_recall


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evalue le modele sur le set de validation."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f2 = fbeta_score(all_labels, all_preds, beta=2, pos_label=1, zero_division=0)

    return epoch_loss, epoch_recall, f2, np.array(all_probs), np.array(all_labels), np.array(all_preds)


print("[OK] Fonctions d'entrainement pretes.")

best_recall = 0.0
history = {"train_loss": [], "val_loss": [], "train_recall": [], "val_recall": [], "val_f2": []}

# --- PHASE 1 : Feature Extraction (backbone gele) ---
print("\n" + "=" * 60)
print("PHASE 1 -- Feature Extraction (backbone gele)")
print("=" * 60)

model.freeze_backbone()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
phase1_epochs = max(NUM_EPOCHS // 3, 3)

for epoch in range(phase1_epochs):
    train_loss, train_recall = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_recall, val_f2, val_probs, val_labels_arr, val_preds = evaluate(model, val_loader, criterion)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_recall"].append(train_recall)
    history["val_recall"].append(val_recall)
    history["val_f2"].append(val_f2)

    print(f"  [P1] Epoch {epoch+1}/{phase1_epochs} | "
          f"Train Loss={train_loss:.4f} Recall={train_recall:.3f} | "
          f"Val Loss={val_loss:.4f} Recall={val_recall:.3f} F2={val_f2:.3f}")

    if val_recall > best_recall:
        best_recall = val_recall
        torch.save(model.state_dict(), "best_model.pt")

# --- PHASE 2 : Fine-Tuning (top layers degeles) ---
print("\n" + "=" * 60)
print("PHASE 2 -- Fine-Tuning (2 derniers layers degeles)")
print("=" * 60)

model.unfreeze_top_layers(n_layers=2)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - phase1_epochs)
phase2_epochs = NUM_EPOCHS - phase1_epochs

for epoch in range(phase2_epochs):
    train_loss, train_recall = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_recall, val_f2, val_probs, val_labels_arr, val_preds = evaluate(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_recall"].append(train_recall)
    history["val_recall"].append(val_recall)
    history["val_f2"].append(val_f2)

    print(f"  [P2] Epoch {epoch+1}/{phase2_epochs} | "
          f"Train Loss={train_loss:.4f} Recall={train_recall:.3f} | "
          f"Val Loss={val_loss:.4f} Recall={val_recall:.3f} F2={val_f2:.3f}")

    if val_recall > best_recall:
        best_recall = val_recall
        torch.save(model.state_dict(), "best_model.pt")

print(f"\nMeilleur rappel en validation : {best_recall:.4f}")

# Recharger le meilleur modele
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE, weights_only=True))
model.eval()

    model, val_loader, criterion
)

# Classification report
print("\n--- Rapport de classification ---")
print(classification_report(val_labels_arr, val_preds, target_names=["OK", "Defaut"]))

# Metriques CND
val_precision = precision_score(val_labels_arr, val_preds, pos_label=1, zero_division=0)
fpr_arr, tpr_arr, _ = roc_curve(val_labels_arr, val_probs)
roc_auc = auc(fpr_arr, tpr_arr)

print(f"Rappel (POD)    : {val_recall:.4f}")
print(f"Precision       : {val_precision:.4f}")
print(f"F2-Score        : {val_f2:.4f}")
print(f"ROC AUC         : {roc_auc:.4f}")

# Seuil optimal (rappel >= 95%)
best_threshold = 0.5
best_prec_at_recall = 0.0
for t in np.arange(0.05, 0.95, 0.01):
    preds_t = (val_probs >= t).astype(int)
    rec_t = recall_score(val_labels_arr, preds_t, pos_label=1, zero_division=0)
    prec_t = precision_score(val_labels_arr, preds_t, pos_label=1, zero_division=0)
    if rec_t >= 0.95 and prec_t > best_prec_at_recall:
        best_prec_at_recall = prec_t
        best_threshold = t

print(f"\nSeuil optimal (rappel >= 95%) : {best_threshold:.2f}")
print(f"  Precision a ce seuil : {best_prec_at_recall:.4f}")

# Courbes d'entrainement - Loss
axes[0, 0].plot(history["train_loss"], label="Train", color="steelblue")
axes[0, 0].plot(history["val_loss"], label="Val", color="coral")
axes[0, 0].axvline(phase1_epochs - 0.5, color="gray", linestyle="--", alpha=0.5, label="Phase 1 -> 2")
axes[0, 0].set_title("Loss par epoch")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].legend()

# Courbes d'entrainement - Rappel
axes[0, 1].plot(history["train_recall"], label="Train", color="steelblue")
axes[0, 1].plot(history["val_recall"], label="Val", color="coral")
axes[0, 1].axvline(phase1_epochs - 0.5, color="gray", linestyle="--", alpha=0.5)
axes[0, 1].set_title("Rappel (POD) par epoch")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].legend()

# F2-Score
axes[0, 2].plot(history["val_f2"], label="Val F2", color="green")
axes[0, 2].axvline(phase1_epochs - 0.5, color="gray", linestyle="--", alpha=0.5)
axes[0, 2].set_title("F2-Score (validation)")
axes[0, 2].set_xlabel("Epoch")
axes[0, 2].legend()

# Matrice de confusion
cm = confusion_matrix(val_labels_arr, val_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["OK", "Defaut"], yticklabels=["OK", "Defaut"], ax=axes[1, 0])
axes[1, 0].set_xlabel("Prediction")
axes[1, 0].set_ylabel("Realite")
axes[1, 0].set_title("Matrice de confusion")

# Courbe ROC
axes[1, 1].plot(fpr_arr, tpr_arr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
axes[1, 1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
axes[1, 1].axhline(y=0.95, color="red", linestyle=":", alpha=0.7, label="POD cible = 95%")
axes[1, 1].set_xlabel("Taux de Faux Positifs")
axes[1, 1].set_ylabel("Taux de Vrais Positifs (POD)")
axes[1, 1].set_title("Courbe ROC")
axes[1, 1].legend(loc="lower right")

# Courbe Precision-Rappel
prec_curve, rec_curve, _ = precision_recall_curve(val_labels_arr, val_probs)
axes[1, 2].plot(rec_curve, prec_curve, color="green", lw=2)
axes[1, 2].axvline(x=0.95, color="red", linestyle=":", alpha=0.7, label="Rappel cible = 95%")
axes[1, 2].set_xlabel("Rappel")
axes[1, 2].set_ylabel("Precision")
axes[1, 2].set_title("Courbe Precision-Rappel")
axes[1, 2].legend()

plt.suptitle("Evaluation du modele -- Controle Qualite", fontsize=14)
plt.tight_layout()
plt.show()

    """Grad-CAM : visualise les zones d'attention du CNN."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.backbone.layer4[-1].conv2
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay(self, image, cam, alpha=0.5):
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


grad_cam = GradCAM(model)

# Trouver des exemples de defauts correctement detectes
transform_inf = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
defect_shown = 0

for path in defect_images[:20]:
    if defect_shown >= 4:
        break
    img = cv2.imread(path)
    processed = preprocess_image(img)
    tensor = transform_inf(image=processed)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    prob_defect = probs[0, 1].item()

    if prob_defect > 0.5:
        cam = grad_cam.generate(tensor, class_idx=1)
        overlay = grad_cam.overlay(processed, cam)

        axes[0, defect_shown].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        axes[0, defect_shown].set_title(f"Original (P={prob_defect:.0%})", fontsize=9)
        axes[0, defect_shown].axis("off")

        axes[1, defect_shown].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, defect_shown].set_title("Grad-CAM", fontsize=9)
        axes[1, defect_shown].axis("off")

        defect_shown += 1

axes[0, 0].set_ylabel("Image", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Grad-CAM", fontsize=11, fontweight="bold")
plt.suptitle("Explicabilite Grad-CAM -- Zones d'attention sur les defauts", fontsize=13)
plt.tight_layout()
plt.show()

print("[OK] Grad-CAM genere.")

    """Prediction sur une image individuelle."""
    img = cv2.imread(image_path)
    processed = preprocess_image(img)
    tensor = transform_inf(image=processed)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)

    prob_ok = probs[0, 0].item()
    prob_defect = probs[0, 1].item()
    is_defect = prob_defect >= threshold

    return {
        "prediction": "DEFAUT" if is_defect else "OK",
        "prob_ok": round(prob_ok, 4),
        "prob_defect": round(prob_defect, 4),
        "threshold": threshold
    }


# Test sur quelques images
print("\n--- Predictions individuelles ---")
test_samples = (ok_images[:3] + defect_images[:3])
for path in test_samples:
    result = predict_single_image(model, path, threshold=best_threshold)
    label_true = "OK" if path in ok_images else "DEFAUT"
    status = "CORRECT" if result["prediction"] == label_true else "ERREUR"
    print(f"  [{status}] Vrai={label_true:6s} | Predit={result['prediction']:6s} | "
          f"P(defaut)={result['prob_defect']:.3f}")

print("\n[OK] Projet 2 termine.")
print("Modele sauvegarde dans best_model.pt")
print("Pour Streamlit, adaptez le code de la cellule Grad-CAM dans une interface.")
