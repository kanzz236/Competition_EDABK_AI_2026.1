import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import (
    SyntheticMNIST1020,
    get_train_transform,
    get_val_transform
)
from model import MNIST1020Net


SEED = 2026


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    seed_everything()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_ds = SyntheticMNIST1020(
        num_samples_per_class=18000,
        transform=get_train_transform(),
        split="train",
        seed=SEED
    )

    val_ds = SyntheticMNIST1020(
        num_samples_per_class=2500,
        transform=get_val_transform(),
        split="val",
        seed=SEED
    )

    workers = min(4, os.cpu_count() or 1)

    batch_size = (
        256 if torch.cuda.is_available()
        else 128
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0
    )

    model = MNIST1020Net(num_classes=11).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.02
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99)
    )

    epochs = 35

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=2e-5
    )

    use_amp = torch.cuda.is_available()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    os.makedirs("./models", exist_ok=True)

    best_val_acc = -1.0
    best_val_loss = float("inf")

    patience = 7
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images = images.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp
            ):
                logits = model(images)
                loss = criterion(
                    logits,
                    targets
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)
            scaler.update()

            train_loss += (
                loss.item() * targets.size(0)
            )

            train_correct += (
                (logits.argmax(1) == targets)
                .sum()
                .item()
            )

            train_total += targets.size(0)

        scheduler.step()

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(
                    device,
                    non_blocking=True
                )

                targets = targets.to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=use_amp
                ):
                    logits = model(images)
                    loss = criterion(
                        logits,
                        targets
                    )

                val_loss += (
                    loss.item() * targets.size(0)
                )

                val_correct += (
                    (logits.argmax(1) == targets)
                    .sum()
                    .item()
                )

                val_total += targets.size(0)

        tr_loss = train_loss / train_total
        tr_acc = train_correct / train_total
        va_loss = val_loss / val_total
        va_acc = val_correct / val_total

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {tr_loss:.4f} "
            f"acc {tr_acc * 100:.2f}% | "
            f"val loss {va_loss:.4f} "
            f"acc {va_acc * 100:.2f}% | "
            f"lr {lr:.2e}"
        )

        improved = (
            va_acc > best_val_acc
            or (
                va_acc == best_val_acc
                and va_loss < best_val_loss
            )
        )

        if improved:
            best_val_acc = va_acc
            best_val_loss = va_loss
            bad_epochs = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "epoch": epoch
                },
                "./models/best_model.pth"
            )

            print("  -> saved best_model.pth")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping.")
            break

    print(
        f"Best validation accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )


if __name__ == "__main__":
    train()