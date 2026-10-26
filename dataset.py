import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


def crop_tight(img, threshold=8):
    ys, xs = np.where(img > threshold)

    if len(xs) == 0:
        return img

    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def affine_digit(img, rng):
    h, w = img.shape
    pad = 4

    canvas = cv2.copyMakeBorder(
        img,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    h2, w2 = canvas.shape

    angle = float(rng.uniform(-8.0, 8.0))
    shear = float(rng.uniform(-0.10, 0.10))
    scale = float(rng.uniform(0.94, 1.06))

    cx, cy = w2 / 2.0, h2 / 2.0

    M = cv2.getRotationMatrix2D(
        (cx, cy),
        angle,
        scale
    )

    M[0, 1] += shear

    out = cv2.warpAffine(
        canvas,
        M,
        (w2, h2),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return crop_tight(out)


def prepare_digit(img, rng):
    img = crop_tight(img)

    if rng.random() < 0.70:
        kernel = np.ones((2, 2), np.uint8)

        if rng.random() < 0.82:
            img = cv2.erode(
                img,
                kernel,
                iterations=1
            )
        else:
            img = cv2.dilate(
                img,
                kernel,
                iterations=1
            )

    img = affine_digit(img, rng)

    target_h = int(rng.integers(16, 22))

    scale = target_h / max(1, img.shape[0])

    target_w = max(
        2,
        int(round(img.shape[1] * scale))
    )

    img = cv2.resize(
        img,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )

    if rng.random() < 0.35:
        sigma = float(rng.uniform(0.15, 0.65))

        img = cv2.GaussianBlur(
            img,
            (0, 0),
            sigma
        )

    gain = float(rng.uniform(0.78, 1.05))

    img = np.clip(
        img.astype(np.float32) * gain,
        0,
        255
    ).astype(np.uint8)

    return img


def make_canvas(img1, img2, rng):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    h = max(h1, h2)

    a = np.zeros((h, w1), dtype=np.uint8)
    b = np.zeros((h, w2), dtype=np.uint8)

    y1 = int(rng.integers(0, h - h1 + 1))
    y2 = int(rng.integers(0, h - h2 + 1))

    a[y1:y1 + h1] = img1
    b[y2:y2 + h2] = img2

    gap = int(rng.integers(-1, 4))

    if gap >= 0:
        combined = np.hstack([
            a,
            np.zeros((h, gap), np.uint8),
            b
        ])
    else:
        overlap = min(-gap, w1, w2)

        combined = np.hstack([
            a[:, :-overlap],
            np.maximum(
                a[:, -overlap:],
                b[:, :overlap]
            ),
            b[:, overlap:]
        ])

    return combined


def fit_to_28(img, rng):
    h, w = img.shape

    if rng.random() < 0.45:
        pad = 2

        tmp = cv2.copyMakeBorder(
            img,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=0
        )

        hh, ww = tmp.shape

        angle = float(rng.uniform(-3.0, 3.0))
        scale = float(rng.uniform(0.96, 1.04))

        M = cv2.getRotationMatrix2D(
            (ww / 2, hh / 2),
            angle,
            scale
        )

        tmp = cv2.warpAffine(
            tmp,
            M,
            (ww, hh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        img = crop_tight(tmp)

    max_side = max(img.shape)

    if max_side > 24:
        s = 24.0 / max_side

        nw = max(
            1,
            int(round(img.shape[1] * s))
        )

        nh = max(
            1,
            int(round(img.shape[0] * s))
        )

        img = cv2.resize(
            img,
            (nw, nh),
            interpolation=cv2.INTER_AREA
        )

    h, w = img.shape
    canvas = np.zeros((28, 28), dtype=np.uint8)

    y = (
        (28 - h) // 2
        + int(rng.integers(-1, 2))
    )

    x = (
        (28 - w) // 2
        + int(rng.integers(-1, 2))
    )

    y = max(0, min(28 - h, y))
    x = max(0, min(28 - w, x))

    canvas[y:y + h, x:x + w] = img

    if rng.random() < 0.20:
        noise = rng.normal(
            0,
            float(rng.uniform(0.5, 2.5)),
            canvas.shape
        )

        canvas = np.clip(
            canvas.astype(np.float32) + noise,
            0,
            255
        ).astype(np.uint8)

    return canvas


class SyntheticMNIST1020(Dataset):
    def __init__(
        self,
        num_samples_per_class=18000,
        transform=None,
        split="train",
        seed=2026
    ):
        self.transform = transform
        self.data = []
        self.targets = []

        os.makedirs(
            "./data/mnist",
            exist_ok=True
        )

        mnist = MNIST(
            root="./data/mnist",
            train=True,
            download=True
        )

        images = mnist.data.numpy()
        labels = mnist.targets.numpy()

        source_rng = np.random.default_rng(seed)
        source_indices = {}

        for d in range(10):
            idx = np.where(labels == d)[0].copy()
            source_rng.shuffle(idx)

            cut = int(len(idx) * 0.90)

            if split == "train":
                source_indices[d] = idx[:cut]
            else:
                source_indices[d] = idx[cut:]

        rng = np.random.default_rng(
            seed + (0 if split == "train" else 100000)
        )

        def generate_pair(d1, d2):
            i1 = int(rng.choice(source_indices[d1]))
            i2 = int(rng.choice(source_indices[d2]))

            img1 = prepare_digit(
                images[i1],
                rng
            )

            img2 = prepare_digit(
                images[i2],
                rng
            )

            combined = make_canvas(
                img1,
                img2,
                rng
            )

            return fit_to_28(combined, rng)

        for cls in range(10, 21):
            d1 = cls // 10
            d2 = cls % 10
            target = cls - 10

            for _ in range(num_samples_per_class):
                self.data.append(
                    generate_pair(d1, d2)
                )
                self.targets.append(target)

        self.data = np.asarray(
            self.data,
            dtype=np.uint8
        )

        self.targets = np.asarray(
            self.targets,
            dtype=np.int64
        )

        order = rng.permutation(len(self.targets))

        self.data = self.data[order]
        self.targets = self.targets[order]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = (
            self.data[idx].astype(np.float32) / 255.0
        )

        if self.transform is not None:
            img = self.transform(img)

        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.tensor(
                self.targets[idx],
                dtype=torch.long
            )
        )


def get_train_transform():
    def transform(img):
        if np.random.rand() < 0.35:
            gain = np.random.uniform(0.92, 1.08)

            img = np.clip(
                img * gain,
                0,
                1
            )

        return img.astype(np.float32)

    return transform


def get_val_transform():
    return None