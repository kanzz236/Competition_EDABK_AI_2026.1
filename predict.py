import os
import numpy as np
import pandas as pd
import torch
from model import MNIST1020Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/best_model.pth"

def shift(img, dx, dy):
    out = np.zeros_like(img)
    xs, xe = max(0, dx), min(28, 28 + dx)
    ys, ye = max(0, dy), min(28, 28 + dy)
    out[ys:ye, xs:xe] = img[ys-dy:ye-dy, xs-dx:xe-dx]
    return out

def preprocess(img):
    img = img.reshape(28, 28).astype(np.float32)
    if img.max() > 1:
        img /= 255.0
    return img

def main():
    print("Device:", DEVICE)

    df = pd.read_csv("test.csv")
    pixels = df.drop(columns=["id"], errors="ignore").values

    print("Test shape:", df.shape)
    print("Pixels:", pixels.shape[1])

    model = MNIST1020Net(11).to(DEVICE)

    ckpt = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print("Checkpoint:", MODEL_PATH)
        print("Epoch:", ckpt.get("epoch"))
        print("Validation accuracy:", ckpt.get("val_acc"))
    else:
        model.load_state_dict(ckpt)

    model.eval()

    imgs = np.stack([preprocess(x) for x in pixels])


    tta = [
        imgs,
        np.stack([shift(x, -1, 0) for x in imgs]),
        np.stack([shift(x, 1, 0) for x in imgs]),
        np.stack([shift(x, 0, -1) for x in imgs]),
        np.stack([shift(x, 0, 1) for x in imgs])
    ]

    preds = []
    probs = []

    with torch.no_grad():
        for i, batch in enumerate(tta):
            x = torch.tensor(
                batch,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(1)

            p = torch.softmax(model(x), dim=1)
            probs.append(p.cpu())

            print(f"TTA {i+1}/5 done")

    prob = torch.stack(probs).mean(0)
    pred = prob.argmax(1).numpy() + 10
    conf = prob.max(1).values.numpy()

    if "id" in df.columns:
        ids = df["id"]
    else:
        ids = np.arange(len(pred))

    submission = pd.DataFrame({
        "id": ids,
        "label": pred
    })

    submission.to_csv("submission.csv", index=False)

    print("\nPrediction distribution:")
    print(pd.Series(pred).value_counts().sort_index())

    print(f"\nMean confidence: {conf.mean():.4f}")
    print(f"Low confidence <0.60: {(conf < 0.60).sum()}")
    print("Saved: submission.csv")

if __name__ == "__main__":
    main()