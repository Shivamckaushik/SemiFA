"""
Encoder Comparison Experiment — Paper Table: Visual Encoder Approaches
=======================================================================
Trains and evaluates three visual encoder baselines on the *same*
SemiFA-930 train/val split used in the paper (train.jsonl 790 images,
val.jsonl 140 images), producing an apples-to-apples comparison for
Table "Comparison of visual encoder approaches" in semifa_arxiv.tex.

Models evaluated
----------------
1. DINOv2-base + MLP head (frozen encoder, 200K trainable params)
2. ResNet-50 end-to-end fine-tuned on SemiFA-930
3. CLIP ViT-B/32 zero-shot (class-name text prompts, 0 trainable params)

Usage
-----
    # From project root (local or Colab after mounting Drive)
    python experiments/run_encoder_comparison.py

    # Override data root if images live elsewhere (e.g. Colab)
    python experiments/run_encoder_comparison.py --data-root /content/drive/MyDrive/SemiFA

    # Quick smoke test (fewer epochs)
    python experiments/run_encoder_comparison.py --epochs 20

Output
------
    experiments/encoder_comparison_results.json
    Printed LaTeX-ready table for copy-paste into the paper.

Notes
-----
- GPU strongly recommended for CLIP and ResNet-50. DINOv2 runs on CPU.
- All three models are evaluated on the identical val.jsonl 140-image set.
- ResNet-50 uses ImageNet pretrained weights + fine-tuned final FC layer only
  for the first 10 epochs, then all layers unfrozen for remaining epochs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModel

# ── Constants ─────────────────────────────────────────────────────────────────

DEFECT_CLASSES = [
    "scratch", "particle_contamination", "edge_crack",
    "center_cluster", "local_cluster", "ring_pattern",
    "random_defects", "near_full_wafer", "no_defect",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(DEFECT_CLASSES)}
NUM_CLASSES = len(DEFECT_CLASSES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP prompt template — more descriptive than bare class name
CLIP_PROMPTS = {
    "scratch":               "a semiconductor wafer map showing a linear scratch defect pattern",
    "particle_contamination":"a semiconductor wafer map with particle contamination defects",
    "edge_crack":            "a semiconductor wafer map showing edge crack defects at the wafer boundary",
    "center_cluster":        "a semiconductor wafer map with a cluster of defects at the center",
    "local_cluster":         "a semiconductor wafer map with a local cluster of defects",
    "ring_pattern":          "a semiconductor wafer map showing a ring-shaped defect pattern",
    "random_defects":        "a semiconductor wafer map with randomly distributed defects",
    "near_full_wafer":       "a semiconductor wafer map with defects covering nearly the full wafer",
    "no_defect":             "a clean semiconductor wafer map with no defects",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path, data_root: Path) -> list[dict]:
    """Load a JSONL annotation file; resolve image paths against data_root."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cls = rec.get("defect_class", "")
            if cls not in CLASS_TO_IDX:
                continue
            # Resolve image path: try relative to data_root first, then absolute
            img_rel = rec.get("image", rec.get("image_path", ""))
            img_path = data_root / img_rel
            if not img_path.exists():
                img_path = Path(img_rel)  # try as absolute/cwd-relative
            if img_path.exists():
                records.append({"image_path": str(img_path), "defect_class": cls})
    return records


def load_image(path: str, transform) -> torch.Tensor | None:
    try:
        return transform(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


# ── Macro F1 (no sklearn dependency) ─────────────────────────────────────────

def macro_f1(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


# ─────────────────────────────────────────────────────────────────────────────
# 1. DINOv2-base + MLP head
# ─────────────────────────────────────────────────────────────────────────────

class DefectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x):
        return self.net(x)


def extract_dino_embeddings(
    records: list[dict],
    processor,
    backbone: nn.Module,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings, labels = [], []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        imgs, valid = [], []
        for r in batch:
            img = load_image(r["image_path"], lambda x: x)  # PIL only
            if img is not None:
                imgs.append(img)
                valid.append(r)
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = backbone(**inputs)
        cls_embs = out.last_hidden_state[:, 0, :].cpu().numpy()
        for emb, rec in zip(cls_embs, valid):
            embeddings.append(emb)
            labels.append(CLASS_TO_IDX[rec["defect_class"]])
        print(f"  DINOv2 embeddings: {min(start+batch_size, len(records))}/{len(records)}", end="\r")
    print()
    return np.array(embeddings, dtype=np.float32), np.array(labels, dtype=np.int64)


def run_dinov2(
    train_records: list[dict],
    val_records: list[dict],
    epochs: int = 50,
) -> dict:
    print("\n[1/3] DINOv2-base + MLP head")
    print("-" * 40)
    t0 = time.time()

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    backbone = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"  Model loaded ({time.time()-t0:.1f}s)")

    print(f"  Extracting train embeddings ({len(train_records)} images)...")
    X_train, y_train = extract_dino_embeddings(train_records, processor, backbone)
    print(f"  Extracting val embeddings ({len(val_records)} images)...")
    X_val, y_val = extract_dino_embeddings(val_records, processor, backbone)

    # Train MLP
    mlp = DefectMLP().to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_train).to(DEVICE)
    yt = torch.tensor(y_train).to(DEVICE)

    # Mini-batch training — matches original Colab training setup
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"  Training MLP ({epochs} epochs, mini-batch 64)...")
    for epoch in range(1, epochs + 1):
        mlp.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            opt.zero_grad()
            loss = crit(mlp(X_batch), y_batch)
            loss.backward()
            opt.step()
        scheduler.step()
        if epoch % 10 == 0:
            mlp.eval()
            with torch.no_grad():
                Xall = torch.tensor(X_train).to(DEVICE)
                yall = torch.tensor(y_train).to(DEVICE)
                acc = (mlp(Xall).argmax(1) == yall).float().mean().item()
            print(f"    epoch {epoch:3d}/{epochs}  train_acc={acc:.1%}")

    # Evaluate on val
    mlp.eval()
    Xv = torch.tensor(X_val).to(DEVICE)
    with torch.no_grad():
        preds = mlp(Xv).argmax(1).cpu().numpy()
    acc = (preds == y_val).mean()
    f1 = macro_f1(preds, y_val, NUM_CLASSES)
    trainable = sum(p.numel() for p in mlp.parameters() if p.requires_grad)

    print(f"  Val accuracy: {acc:.1%}  Macro F1: {f1:.3f}  "
          f"Trainable params: {trainable/1e3:.0f}K")
    return {
        "method": "DINOv2-base + MLP (ours)",
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "trainable_params": trainable,
        "n_val": len(y_val),
        "runtime_s": time.time() - t0,
        "note": "Frozen encoder + lightweight MLP head trained on SemiFA-930 train split",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. ResNet-50 end-to-end fine-tuned
# ─────────────────────────────────────────────────────────────────────────────

RESNET_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
RESNET_TRANSFORM_VAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SemiDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[r["defect_class"]]


def run_resnet50(
    train_records: list[dict],
    val_records: list[dict],
    epochs: int = 30,
    batch_size: int = 32,
) -> dict:
    print("\n[2/3] ResNet-50 end-to-end fine-tuned")
    print("-" * 40)
    t0 = time.time()

    # Load pretrained ResNet-50, replace FC
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    train_ds = SemiDataset(train_records, RESNET_TRANSFORM_TRAIN)
    val_ds   = SemiDataset(val_records,   RESNET_TRANSFORM_VAL)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    crit = nn.CrossEntropyLoss()

    # Phase 1 (epochs 1–10): freeze backbone, train FC only
    for name, param in model.named_parameters():
        param.requires_grad = (name.startswith("fc"))
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # Phase 2 (epochs 11–end): unfreeze all layers with lower LR
    warmup_epochs = min(10, epochs // 3)
    full_epochs   = epochs - warmup_epochs

    print(f"  Phase 1: FC-only warmup ({warmup_epochs} epochs)...")
    for epoch in range(1, warmup_epochs + 1):
        model.train()
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            crit(model(imgs), lbls).backward()
            opt.step()
        if epoch % 5 == 0 or epoch == warmup_epochs:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, lbls in val_dl:
                    preds = model(imgs.to(DEVICE)).argmax(1).cpu()
                    correct += (preds == lbls).sum().item()
                    total   += len(lbls)
            print(f"    epoch {epoch:3d}/{warmup_epochs}  val_acc={correct/total:.1%}")

    print(f"  Phase 2: Full fine-tune ({full_epochs} epochs)...")
    for param in model.parameters():
        param.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=full_epochs)

    for epoch in range(1, full_epochs + 1):
        model.train()
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            crit(model(imgs), lbls).backward()
            opt.step()
        scheduler.step()
        if epoch % 5 == 0 or epoch == full_epochs:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, lbls in val_dl:
                    preds = model(imgs.to(DEVICE)).argmax(1).cpu()
                    correct += (preds == lbls).sum().item()
                    total   += len(lbls)
            print(f"    epoch {epoch:3d}/{full_epochs}  val_acc={correct/total:.1%}")

    # Final evaluation with macro F1
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in val_dl:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    f1  = macro_f1(all_preds, all_labels, NUM_CLASSES)
    trainable = sum(p.numel() for p in model.parameters())

    print(f"  Val accuracy: {acc:.1%}  Macro F1: {f1:.3f}  "
          f"Total params: {trainable/1e6:.1f}M")
    return {
        "method": "ResNet-50 (end-to-end, SemiFA-930)",
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "trainable_params": trainable,
        "n_val": len(all_labels),
        "runtime_s": time.time() - t0,
        "note": "ImageNet pretrained weights, FC replaced, trained on SemiFA-930 train split",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLIP ViT-B/32 zero-shot
# ─────────────────────────────────────────────────────────────────────────────

def run_clip_zeroshot(val_records: list[dict]) -> dict:
    print("\n[3/3] CLIP ViT-B/32 zero-shot")
    print("-" * 40)
    t0 = time.time()

    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("  [SKIP] transformers CLIP not available. "
              "Install: pip install transformers>=4.30")
        return {
            "method": "CLIP ViT-B/32 zero-shot",
            "accuracy": None,
            "macro_f1": None,
            "trainable_params": 0,
            "n_val": len(val_records),
            "note": "SKIPPED — transformers not available",
        }

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print(f"  Model loaded ({time.time()-t0:.1f}s)")

    # Pre-encode all class text prompts
    text_inputs = clip_proc(
        text=list(CLIP_PROMPTS.values()),
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        # Use model components directly — get_text_features() returns
        # BaseModelOutputWithPooling in some transformers versions
        text_out = clip_model.text_model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
        )
        text_features = clip_model.text_projection(text_out.pooler_output)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    all_preds, all_labels = [], []
    batch_size = 32

    for start in range(0, len(val_records), batch_size):
        batch = val_records[start:start + batch_size]
        imgs, valid = [], []
        for r in batch:
            img = load_image(r["image_path"], lambda x: x)
            if img is not None:
                imgs.append(img)
                valid.append(r)
        if not imgs:
            continue
        img_inputs = clip_proc(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_out = clip_model.vision_model(pixel_values=img_inputs['pixel_values'])
            img_features = clip_model.visual_projection(img_out.pooler_output)
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
            sims = (img_features @ text_features.T)
            preds = sims.argmax(dim=-1).cpu().numpy()
        for pred, rec in zip(preds, valid):
            all_preds.append(int(pred))
            all_labels.append(CLASS_TO_IDX[rec["defect_class"]])
        print(f"  CLIP inference: {min(start+batch_size, len(val_records))}/{len(val_records)}", end="\r")
    print()

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    f1  = macro_f1(all_preds, all_labels, NUM_CLASSES)

    print(f"  Val accuracy: {acc:.1%}  Macro F1: {f1:.3f}  Trainable params: 0")
    return {
        "method": "CLIP ViT-B/32 zero-shot",
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "trainable_params": 0,
        "n_val": len(all_labels),
        "runtime_s": time.time() - t0,
        "note": "Zero-shot — class-specific text prompts, no fine-tuning",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("ENCODER COMPARISON — semifa_arxiv.tex Table (Comparison of Encoders)")
    print("=" * 72)
    print(f"{'Method':<38} {'Accuracy':>9}  {'Macro F1':>9}  {'Params':>10}")
    print("-" * 72)
    for r in results:
        acc = f"{r['accuracy']:.1%}" if r['accuracy'] is not None else "N/A"
        f1  = f"{r['macro_f1']:.3f}"  if r['macro_f1']  is not None else "N/A"
        if r['trainable_params'] == 0:
            params = "0 (zero-shot)"
        elif r['trainable_params'] < 1e6:
            params = f"{r['trainable_params']/1e3:.0f}K"
        else:
            params = f"{r['trainable_params']/1e6:.1f}M"
        print(f"  {r['method']:<36} {acc:>9}  {f1:>9}  {params:>10}")
    print("=" * 72)

    print("\nLaTeX table rows (replace existing tab:encoder_comparison in paper):")
    print("\\midrule")
    for r in results:
        acc = f"\\sim{r['accuracy']:.0%}" if "CLIP" in r["method"] else f"{r['accuracy']:.1%}"
        f1  = f"{r['macro_f1']:.3f}" if r['macro_f1'] is not None else "---"
        if r['trainable_params'] == 0:
            params = "0"
        elif r['trainable_params'] < 1e6:
            params = f"{r['trainable_params']//1000}K"
        else:
            params = f"{r['trainable_params']/1e6:.1f}M"
        method = r['method'].replace("(ours)", "\\textbf{(ours)}")
        print(f"  {method} & {r['accuracy']:.1%} & {f1} & {params} \\\\")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Encoder comparison experiment")
    parser.add_argument(
        "--data-root", type=str, default=".",
        help="Project root directory (default: current working directory). "
             "On Colab: /content/drive/MyDrive/SemiFA"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs for DINOv2-MLP and ResNet-50 (default: 50)"
    )
    parser.add_argument(
        "--skip-clip", action="store_true",
        help="Skip CLIP zero-shot evaluation"
    )
    parser.add_argument(
        "--skip-resnet", action="store_true",
        help="Skip ResNet-50 fine-tuning"
    )
    parser.add_argument(
        "--skip-dinov2", action="store_true",
        help="Skip DINOv2+MLP (use when re-running only CLIP or ResNet-50)"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_jsonl = data_root / "data/processed/train.jsonl"
    val_jsonl   = data_root / "data/processed/val.jsonl"

    # Verify files exist
    for p in [train_jsonl, val_jsonl]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            print("  Make sure --data-root points to the project root containing data/processed/")
            return

    print(f"Device       : {DEVICE}")
    print(f"Data root    : {data_root.resolve()}")
    print(f"Train JSONL  : {train_jsonl}  ", end="")
    train_records = load_jsonl(train_jsonl, data_root)
    print(f"({len(train_records)} records loaded)")

    print(f"Val JSONL    : {val_jsonl}  ", end="")
    val_records = load_jsonl(val_jsonl, data_root)
    print(f"({len(val_records)} records loaded)")

    if len(train_records) < 50 or len(val_records) < 10:
        print("ERROR: Too few records. Verify image paths are correct.")
        return

    results = []

    # 1. DINOv2
    if not args.skip_dinov2:
        results.append(run_dinov2(train_records, val_records, epochs=args.epochs))
    else:
        print("\n[1/3] DINOv2+MLP  — SKIPPED (--skip-dinov2)")

    # 2. ResNet-50
    if not args.skip_resnet:
        results.append(run_resnet50(train_records, val_records, epochs=args.epochs))
    else:
        print("\n[2/3] ResNet-50  — SKIPPED (--skip-resnet)")

    # 3. CLIP
    if not args.skip_clip:
        results.append(run_clip_zeroshot(val_records))
    else:
        print("\n[3/3] CLIP ViT-B/32  — SKIPPED (--skip-clip)")

    # Print summary table
    print_table(results)

    # Save results
    out = {
        "experiment": "SemiFA encoder comparison",
        "dataset": "SemiFA-930",
        "train_split": len(train_records),
        "val_split": len(val_records),
        "device": DEVICE,
        "epochs": args.epochs,
        "results": results,
    }
    out_path = data_root / "experiments/encoder_comparison_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print("\nNext step: copy the LaTeX rows above into paper/semifa_arxiv.tex "
          "tab:encoder_comparison, removing the † footnote.")


if __name__ == "__main__":
    main()
