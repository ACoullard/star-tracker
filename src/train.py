"""
Training entry point for SIFTERN.

Usage
-----
    python src/train.py
    python src/train.py --data data/clean-de-0-6.json --epochs 20

The script:
  1. Loads the dataset and builds the star-ID → class-index mapping.
  2. Splits data 90/10 into train and validation sets.
  3. Trains SIFTERN with AdamW + linear warmup + ReduceLROnPlateau.
  4. Saves the best checkpoint (by validation loss) to `--checkpoint-dir`.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import StarTrackerDataset, collate_fn
from model import SIFTERN

warnings.filterwarnings("ignore", message=".*nested tensor.*", module="torch")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SIFTERN")
    parser.add_argument(
        "--data",
        default="data/clean-de-0-6.json",
        help="Path to the star scene JSON dataset",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=7000)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Sensor side length in pixels (used to locate the image center)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--catalog",
        default="data/filter-catalog.csv",
        help="Path to filter-catalog.csv",
    )
    parser.add_argument(
        "--dec-range",
        type=float,
        nargs=2,
        metavar=("MIN_DEG", "MAX_DEG"),
        required=True,
        help="Declination bounds of the sky partition, e.g. --dec-range 0 6",
    )
    parser.add_argument(
        "--half-fov",
        type=float,
        default=6.0,
        help="Half the camera FOV in degrees (default 6.0 for a 12° FOV)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the best checkpoint in --checkpoint-dir",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup_steps: int):
    """Linear warmup from 0 to 1 over `warmup_steps` steps, constant after."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return lr_lambda


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: SIFTERN,
    loader: DataLoader,
    optimizer: AdamW,
    warmup_scheduler: LambdaLR,
    device: torch.device,
    warmup_steps: int,
    global_step: list[int],
) -> float:
    model.train()
    total_loss = 0.0

    for padded, mask, labels in tqdm(loader, desc="  train", leave=False):
        padded = padded.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(padded, mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        # Advance warmup scheduler while still in warmup phase
        if global_step[0] < warmup_steps:
            warmup_scheduler.step()
        global_step[0] += 1

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model: SIFTERN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean_loss, top1_accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for padded, mask, labels in tqdm(loader, desc="  val  ", leave=False):
        padded = padded.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits, _ = model(padded, mask)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    mean_loss = total_loss / len(loader)
    accuracy = correct / total
    return mean_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: SIFTERN,
    optimizer: AdamW,
    warmup_scheduler: LambdaLR,
    plateau_scheduler: ReduceLROnPlateau,
    epoch: int,
    global_step: int,
    val_loss: float,
    dataset: StarTrackerDataset,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
            "plateau_scheduler_state_dict": plateau_scheduler.state_dict(),
            "star_id_to_idx": dataset.star_id_to_idx,
            "idx_to_star_id": dataset.idx_to_star_id,
            "n_classes": dataset.n_classes,
            "catalog_path": str(dataset._catalog_path),
            "dec_range": dataset._dec_range,
            "half_fov": dataset._half_fov,
            "val_loss": val_loss,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset ---
    print("Loading dataset...")
    dec_range = tuple(args.dec_range)
    dataset = StarTrackerDataset(
        path=args.data,
        catalog_path=args.catalog,
        dec_range=dec_range,
        half_fov=args.half_fov,
        image_size=args.image_size,
    )
    print(f"  Samples : {len(dataset):,}")
    print(f"  Classes : {dataset.n_classes}")

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    model = SIFTERN(n_classes=dataset.n_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params  : {total_params:,}")

    # --- Optimizer + schedulers ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=make_lr_lambda(args.warmup_steps))
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    # --- Checkpoint dir ---
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"

    # --- Resume ---
    best_val_loss = float("inf")
    global_step = [0]
    start_epoch = 1

    if args.resume and best_checkpoint.exists():
        ckpt = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "warmup_scheduler_state_dict" in ckpt:
            warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state_dict"])
        if "plateau_scheduler_state_dict" in ckpt:
            plateau_scheduler.load_state_dict(ckpt["plateau_scheduler_state_dict"])
        global_step[0] = ckpt.get("global_step", 0)
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(
            f"  Resumed from epoch {ckpt['epoch']}"
            f"  (global_step={global_step[0]}, val_loss={best_val_loss:.4f})"
        )
    elif args.resume:
        print("  --resume set but no checkpoint found, starting from scratch")

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, warmup_scheduler,
            device, args.warmup_steps, global_step,
        )
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        # Only hand control to plateau scheduler after warmup
        if global_step[0] >= args.warmup_steps:
            plateau_scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  val_acc={val_acc*100:.2f}%  lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_checkpoint, model, optimizer,
                warmup_scheduler, plateau_scheduler,
                epoch, global_step[0], val_loss, dataset,
            )
            print(f"  Saved best checkpoint → {best_checkpoint}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
