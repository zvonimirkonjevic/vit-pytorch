"""Trains a ViT-Base/16 from scratch on the UC Merced Land Use dataset and saves its loss curves.

Expects data/train and data/val, created by scripts/prepare_data.py.

Usage:
    uv run scripts/train.py --epochs 10 --batch-size 32 --lr 3e-3
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

from vit.dataset import create_dataloaders
from vit.model import ViT
from vit.train import train

REPO_ROOT = Path(__file__).resolve().parents[1]

# Palette taken from the README GIFs (class token purple, patch embedding pink, light gray guides)
TRAIN_COLOR = "#9575cd"
VAL_COLOR = "#e8829b"
GUIDE_COLOR = "#bfbfbf"


def set_seed(seed: int = 42):
    """Seeds PyTorch's CPU and CUDA random number generators for reproducible runs.

    Args:
        seed: Seed value passed to the generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def plot_loss_curves(results: dict[str, list[float]], plot_path: Path):
    """Plots training and validation loss per epoch and saves the figure.

    Args:
        results: Metrics returned by vit.train.train().
        plot_path: Path of the saved PNG.
    """
    epochs_range = range(1, len(results["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    for key, label, color in [("train_loss", "train", TRAIN_COLOR), ("test_loss", "val", VAL_COLOR)]:
        ax.plot(epochs_range, results[key], color=color, linewidth=2, marker="o", markersize=4)
        # Label each curve at its last point instead of a legend box to keep the plot uncluttered
        ax.annotate(label, xy=(epochs_range[-1], results[key][-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=color, fontsize=10)

    ax.set_title("Loss", fontsize=13, pad=12)
    ax.set_xlabel("Epoch", color="#555555")
    ax.set_xticks(list(epochs_range))
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GUIDE_COLOR)
    ax.tick_params(colors="#555555", length=0)
    fig.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    # 8 x 4.5 inches at 160 dpi gives 1280 px, the width of the README GIFs
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train ViT-Base/16 on the UC Merced Land Use dataset.")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data", help="Directory with the train and val folders.")
    parser.add_argument("--image-size", type=int, default=256, help="Side length H = W the images are resized to.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-3, help="Peak learning rate reached after warmup.")
    parser.add_argument("--weight-decay", type=float, default=0.3)
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Epochs of linear learning rate warmup.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-path", type=Path, default=REPO_ROOT / "assets" / "loss_curves.png")
    args = parser.parse_args()
    if args.warmup_epochs >= args.epochs:
        parser.error(f"--warmup-epochs must be smaller than --epochs: warmup_epochs={args.warmup_epochs} epochs={args.epochs}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: device={device}")

    # Resizing replaces dropping the off-size images, so every image splits into the same N patches
    manual_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    train_dataloader, val_dataloader, class_names = create_dataloaders(
        train_dir=args.data_path / "train",
        test_dir=args.data_path / "val",
        transform=manual_transforms,
        batch_size=args.batch_size,
    )

    set_seed(args.seed)
    model = ViT(
        image_size=args.image_size,
        patch_size=16,
        num_classes=len(class_names),
        embedding_dim=768,
        num_transformer_layers=12,
        num_heads=12,
        mlp_dim=3072,
        mlp_dropout=0.1,
        embedding_dropout=0.1,
    ).to(device)

    # Scheduler steps once per batch, so schedule lengths are in optimizer steps, not epochs
    total_steps = args.epochs * len(train_dataloader)
    warmup_steps = args.warmup_epochs * len(train_dataloader)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()
    # LinearLR requires start_factor > 0, so warmup starts from a tiny fraction of the peak lr instead of 0
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=total_steps - warmup_steps
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps]
    )

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
        lr_scheduler=lr_scheduler,
    )

    plot_loss_curves(results, args.plot_path)
    print(f"Saved loss curves: plot_path={args.plot_path}")


if __name__ == "__main__":
    main()
