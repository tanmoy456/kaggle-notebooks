## Brain MRI Classification (Tumor vs No Tumor)

A computer vision project for binary brain MRI classification using PyTorch and transfer learning (ResNet18), with leakage-safe data splitting.

### Project Goal
Build a reliable MRI classifier for:
- `yes` -> Tumor
- `no` -> No Tumor

---

### Key Highlights
- Transfer learning with `ResNet18` (ImageNet pretrained)
- Two-stage training:
  1. Train classifier head
  2. Fine-tune full model with lower learning rate
- Leakage-safe split for augmented data (grouped by original image)
- Evaluation with:
  - Accuracy
  - Confusion Matrix
  - Precision / Recall / F1

---

### Folder Structure
```text
brain-mri-classification/
├── brain_tumor_dataset/
│   ├── yes/
│   └── no/
├── augmented/
│   ├── yes/
│   └── no/
├── dataset_split/
│   ├── train/
│   │   ├── yes/
│   │   └── no/
│   ├── val/
│   │   ├── yes/
│   │   └── no/
│   └── test/
│       ├── yes/
│       └── no/

