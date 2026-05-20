"""
Core utilities for subliminal-learning teacher/student experiments.

- flexible split-head MLP/CNN architectures;
- stable layer names: fc1, fc2, ..., conv1, conv2, ..., class_head, aux_head;
- shared initialisation sources A/B/random;
- per-layer trainability;
- per-layer perturbations at named experiment timings;
- append-safe tidy metric files
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


SCHEMA_VERSION = "subliminal_unified_v1"
EPS = 1e-12

ALLOWED_INIT_SOURCES = {"A", "B", "random"}
ALLOWED_TIMINGS = {
    "before_teacher_training",
    "after_teacher_training",
    "before_student_training",
    "after_student_training",
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    data_dir: str = "./MNIST_DATA"
    class_count: int | None = None
    class_selection: str = "first"
    outdir: str = "./outputs"
    emnist_split: str = "balanced"
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"

    teacher_type: str = "mlp"
    student_type: str = "mlp"
    teacher_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    student_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    teacher_arch_spec: list[dict[str, Any]] | None = None
    student_arch_spec: list[dict[str, Any]] | None = None
    m: int = 10

    teacher_init: str | None = None
    student_init: str | None = None
    teacher_trainable: str | None = None
    student_trainable: str | None = None

    teacher_epochs: int = 5
    student_epochs: int = 5
    data_bsize: int = 1024
    noise_bsize: int = 1000
    noise_steps: int = 60
    teacher_lr: float = 1e-3
    student_lr: float = 1e-3

    noise_dist: str = "uniform"
    perlin_res: int = 8
    normalize_noise: bool = False
    eval_noise_batches: int = 25
    eval_noise_bsize: int = 1000

    perturb_specs: list[dict[str, Any]] = field(default_factory=list)
    sweep_name: str = "default"
    run_label: str = ""
    append_global: bool = False


@dataclass
class ModelBundle:
    model: nn.Module
    layer_names: list[str]
    init_config: dict[str, str]
    trainable_config: dict[str, bool]
    arch_description: dict[str, Any]


# -----------------------------------------------------------------------------
# Reproducibility and parsing helpers
# -----------------------------------------------------------------------------

def stable_int_hash(text: str, modulo: int = 1_000_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def rng_base(seed: int) -> int:
    return 100_000 * int(seed)


def derived_seed(seed: int, *parts: Any) -> int:
    label = ":".join(str(p) for p in parts)
    return rng_base(seed) + stable_int_hash(label, modulo=900_000_000)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def fork_torch_rng(device: torch.device | None = None):
    devices: list[int] = []
    if device is not None and device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        yield


def parse_int_list(value: str | Iterable[int] | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def parse_bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def ordered_unique(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def parse_key_value_config(
    config: str | None,
    layer_names: list[str],
    default: Any,
    parser,
    allowed: set[str] | None = None,
    config_name: str = "config",
) -> dict[str, Any]:
    """Parse positional or named per-layer configs.

    Positional: "A,A,random,A" for layers [fc1, fc2, class_head, aux_head]
    Named:      "fc1:A,fc2:A,class_head:random,aux_head:A"
    Named form supports all:<value> as a default override.
    """
    if config is None or not str(config).strip():
        return {name: default for name in layer_names}

    raw = str(config).strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]

    if any(":" in p for p in parts):
        result = {name: default for name in layer_names}
        seen: set[str] = set()
        for part in parts:
            if ":" not in part:
                raise ValueError(
                    f"Mixed positional/named syntax in {config_name}: {raw!r}. "
                    "Use either all positional entries or key:value entries."
                )
            key, value = [x.strip() for x in part.split(":", 1)]
            parsed = parser(value)
            if allowed is not None and str(parsed) not in allowed:
                raise ValueError(f"Invalid value {parsed!r} in {config_name}; allowed={sorted(allowed)}")
            if key == "all":
                result = {name: parsed for name in layer_names}
                continue
            if key not in layer_names:
                raise ValueError(
                    f"Unknown layer {key!r} in {config_name}. Available layers: {layer_names}"
                )
            result[key] = parsed
            seen.add(key)
        return result

    if len(parts) != len(layer_names):
        raise ValueError(
            f"{config_name} must contain {len(layer_names)} entries for layers {layer_names}; "
            f"got {len(parts)} entries: {parts}"
        )

    values = []
    for value in parts:
        parsed = parser(value)
        if allowed is not None and str(parsed) not in allowed:
            raise ValueError(f"Invalid value {parsed!r} in {config_name}; allowed={sorted(allowed)}")
        values.append(parsed)
    return dict(zip(layer_names, values))


def parse_init_source(value: Any) -> str:
    v = str(value).strip()
    if v.lower() == "random":
        return "random"
    v = v.upper()
    if v in {"A", "B"}:
        return v
    raise ValueError(f"Invalid init source {value!r}; use A, B, or random")


def parse_json_or_path(value: str | None) -> Any:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    path = Path(text)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(text)


def sanitize_filename(value: Any) -> str:
    s = str(value)
    for ch in ["/", "\\", " ", ",", ":", "=", "[", "]", "{", "}", '"', "'"]:
        s = s.replace(ch, "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "run"


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class SplitHeadMLP(nn.Module):
    def __init__(self, hidden_dims: list[int], m: int, num_classes: int):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("MLP needs at least one hidden dimension.")
        dims = [28 * 28] + [int(d) for d in hidden_dims]
        self.hidden_layer_names: list[str] = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            name = f"fc{i}"
            setattr(self, name, nn.Linear(din, dout))
            self.hidden_layer_names.append(name)
        self.class_head = nn.Linear(dims[-1], num_classes)
        self.aux_head = nn.Linear(dims[-1], m)
        self.layer_names = self.hidden_layer_names + ["class_head", "aux_head"]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for name in self.hidden_layer_names:
            x = F.relu(getattr(self, name)(x))
        return x

    def forward_split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        return self.class_head(h), self.aux_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, z = self.forward_split(x)
        return torch.cat([y, z], dim=1)


class SplitHeadCNN(nn.Module):
    """Configurable CNN feature extractor with split class/aux heads.

    Supported feature spec entries:
    - conv2d: out_channels, kernel_size, stride, padding, dilation, activation, pool
    - linear: out_features, activation

    The first linear layer automatically receives the flattened feature dimension.
    """

    def __init__(
        self,
        feature_spec: list[dict[str, Any]],
        m: int,
        num_classes: int,
        input_shape: tuple[int, int, int] = (1, 28, 28),
    ):
        super().__init__()
        if not feature_spec:
            raise ValueError("CNN feature_spec may not be empty.")
        self.feature_spec: list[dict[str, Any]] = []
        self.feature_layer_names: list[str] = []
        in_channels = input_shape[0]
        linear_in_features: int | None = None
        conv_count = 0
        fc_count = 0
        flattened = False

        # Track spatial dimensions analytically for common conv/pool choices.
        h, w = input_shape[1], input_shape[2]

        for raw in feature_spec:
            spec = dict(raw)
            typ = str(spec.get("type", "")).lower()
            if typ == "conv":
                typ = "conv2d"
            if typ not in {"conv2d", "linear"}:
                raise ValueError(f"Unsupported CNN layer type: {typ!r}")

            if typ == "conv2d":
                conv_count += 1
                name = str(spec.get("name") or f"conv{conv_count}")
                out_channels = int(spec["out_channels"])
                kernel_size = spec.get("kernel_size", 3)
                stride = spec.get("stride", 1)
                padding = spec.get("padding", 0)
                dilation = spec.get("dilation", 1)
                bias = parse_bool_value(spec.get("bias", True))
                layer = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                )
                setattr(self, name, layer)
                in_channels = out_channels
                h, w = conv2d_output_hw(h, w, kernel_size, stride, padding, dilation)
                pool = spec.get("pool")
                if pool:
                    h, w = pool_output_hw(h, w, pool)
                spec["name"] = name
                spec["type"] = "conv2d"
                self.feature_spec.append(spec)
                self.feature_layer_names.append(name)
                continue

            # linear layer
            fc_count += 1
            name = str(spec.get("name") or f"fc{fc_count}")
            out_features = int(spec["out_features"])
            if not flattened:
                linear_in_features = in_channels * h * w
                flattened = True
            assert linear_in_features is not None
            layer = nn.Linear(linear_in_features, out_features, bias=parse_bool_value(spec.get("bias", True)))
            setattr(self, name, layer)
            linear_in_features = out_features
            spec["name"] = name
            spec["type"] = "linear"
            self.feature_spec.append(spec)
            self.feature_layer_names.append(name)

        if linear_in_features is None:
            linear_in_features = in_channels * h * w
        self.class_head = nn.Linear(linear_in_features, num_classes)
        self.aux_head = nn.Linear(linear_in_features, m)
        self.layer_names = self.feature_layer_names + ["class_head", "aux_head"]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for spec in self.feature_spec:
            name = spec["name"]
            typ = spec["type"]
            if typ == "conv2d":
                x = getattr(self, name)(x)
                x = apply_activation(x, spec.get("activation", "relu"))
                x = apply_pool(x, spec.get("pool"))
            elif typ == "linear":
                if x.ndim > 2:
                    x = x.view(x.size(0), -1)
                x = getattr(self, name)(x)
                x = apply_activation(x, spec.get("activation", "relu"))
            else:
                raise RuntimeError(f"Unsupported layer type during forward: {typ}")
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return x

    def forward_split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        return self.class_head(h), self.aux_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, z = self.forward_split(x)
        return torch.cat([y, z], dim=1)


def as_pair(x: Any) -> tuple[int, int]:
    if isinstance(x, (list, tuple)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def conv2d_output_hw(h: int, w: int, kernel_size: Any, stride: Any, padding: Any, dilation: Any) -> tuple[int, int]:
    kh, kw = as_pair(kernel_size)
    sh, sw = as_pair(stride)
    ph, pw = as_pair(padding)
    dh, dw = as_pair(dilation)
    oh = math.floor((h + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
    ow = math.floor((w + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
    return oh, ow


def pool_output_hw(h: int, w: int, pool: dict[str, Any]) -> tuple[int, int]:
    kernel_size = pool.get("kernel_size", 2)
    stride = pool.get("stride", kernel_size)
    padding = pool.get("padding", 0)
    dilation = pool.get("dilation", 1)
    return conv2d_output_hw(h, w, kernel_size, stride, padding, dilation)


def apply_activation(x: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None:
        return x
    a = str(activation).lower()
    if a in {"none", "identity", "linear"}:
        return x
    if a == "relu":
        return F.relu(x)
    if a == "gelu":
        return F.gelu(x)
    if a == "tanh":
        return torch.tanh(x)
    if a == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(f"Unsupported activation: {activation!r}")


def apply_pool(x: torch.Tensor, pool: dict[str, Any] | None) -> torch.Tensor:
    if not pool:
        return x
    typ = str(pool.get("type", "max")).lower()
    kernel_size = pool.get("kernel_size", 2)
    stride = pool.get("stride", kernel_size)
    padding = pool.get("padding", 0)
    if typ in {"max", "maxpool", "maxpool2d"}:
        return F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    if typ in {"avg", "average", "avgpool", "avgpool2d"}:
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    if typ in {"adaptive_avg", "adaptive_average"}:
        output_size = pool.get("output_size", 1)
        return F.adaptive_avg_pool2d(x, output_size=output_size)
    raise ValueError(f"Unsupported pool type: {typ!r}")


def default_cnn_spec(latent_dim: int = 256) -> list[dict[str, Any]]:
    return [
        {
            "name": "conv1",
            "type": "conv2d",
            "out_channels": 32,
            "kernel_size": 3,
            "padding": 1,
            "activation": "relu",
            "pool": {"type": "max", "kernel_size": 2},
        },
        {
            "name": "conv2",
            "type": "conv2d",
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "activation": "relu",
            "pool": {"type": "max", "kernel_size": 2},
        },
        {
            "name": "fc1",
            "type": "linear",
            "out_features": int(latent_dim),
            "activation": "relu",
        },
    ]


def make_model(
    model_type: str,
    hidden_dims: list[int],
    arch_spec: list[dict[str, Any]] | None,
    m: int,
    num_classes: int,
    seed: int,
) -> tuple[nn.Module, list[str], dict[str, Any]]:
    set_seed(seed)
    typ = model_type.lower()
    if typ == "mlp":
        model = SplitHeadMLP(hidden_dims=hidden_dims, m=m, num_classes=num_classes)
        desc = {"type": "mlp", "hidden_dims": hidden_dims}
        return model, model.layer_names, desc
    if typ == "cnn":
        spec = arch_spec if arch_spec is not None else default_cnn_spec(hidden_dims[-1] if hidden_dims else 256)
        model = SplitHeadCNN(feature_spec=spec, m=m, num_classes=num_classes)
        desc = {"type": "cnn", "feature_spec": spec}
        return model, model.layer_names, desc
    raise ValueError(f"Unknown model_type: {model_type!r}")


@torch.no_grad()
def reset_layer_from_source_(layer: nn.Module, source: str, layer_name: str, seed: int) -> None:
    if source not in {"A", "B"}:
        raise ValueError(f"Can only reset from A/B sources, got {source!r}")
    layer_seed = derived_seed(seed, "init_source", source, layer_name)
    with fork_torch_rng(None):
        torch.manual_seed(layer_seed)
        layer.reset_parameters()


@torch.no_grad()
def apply_init_config(model: nn.Module, layer_names: list[str], init_config: dict[str, str], seed: int) -> None:
    for layer_name in layer_names:
        source = init_config[layer_name]
        if source in {"A", "B"}:
            reset_layer_from_source_(getattr(model, layer_name), source, layer_name, seed)
        elif source == "random":
            continue
        else:
            raise ValueError(f"Unknown init source for {layer_name}: {source!r}")


def build_model_bundle(
    model_name: Literal["teacher", "student"],
    model_type: str,
    hidden_dims: list[int],
    arch_spec: list[dict[str, Any]] | None,
    init_config_raw: str | None,
    trainable_config_raw: str | None,
    m: int,
    num_classes: int,
    seed: int,
    device: torch.device,
) -> ModelBundle:
    model_seed = derived_seed(seed, model_name, "model_random_init")
    model, layer_names, arch_description = make_model(
        model_type=model_type,
        hidden_dims=hidden_dims,
        arch_spec=arch_spec,
        m=m,
        num_classes=num_classes,
        seed=model_seed,
    )
    init_config = parse_key_value_config(
        init_config_raw,
        layer_names,
        default="A",
        parser=parse_init_source,
        allowed=ALLOWED_INIT_SOURCES,
        config_name=f"{model_name}_init",
    )
    trainable_config = parse_key_value_config(
        trainable_config_raw,
        layer_names,
        default=True,
        parser=parse_bool_value,
        config_name=f"{model_name}_trainable",
    )
    apply_init_config(model, layer_names, init_config, seed=seed)
    return ModelBundle(
        model=model.to(device),
        layer_names=layer_names,
        init_config=init_config,
        trainable_config=trainable_config,
        arch_description=arch_description,
    )


# -----------------------------------------------------------------------------
# Data and noise
# -----------------------------------------------------------------------------

def dataset_num_classes(dataset: str, emnist_split: str = "balanced") -> int:
    name = dataset.lower()
    if name == "mnist":
        return 10
    if name == "emnist":
        if emnist_split == "balanced":
            return 47
        if emnist_split == "letters":
            return 26
        raise ValueError(f"Unknown EMNIST split: {emnist_split!r}")
    raise ValueError(f"Unknown dataset: {dataset!r}")


def emnist_letters_target_transform(y: int) -> int:
    return y - 1

class ClassTruncatedDataset(Dataset):
    """Keep selected classes and remap labels to 0..K-1.

    Filtering is performed after the base dataset target_transform. This means
    EMNIST letters labels are converted from 1..26 to 0..25 before truncation.
    """

    def __init__(self, base: Dataset, keep_classes: list[int]):
        self.base = base
        self.keep_classes = [int(c) for c in keep_classes]
        self.label_map = {old: new for new, old in enumerate(self.keep_classes)}
        keep_set = set(self.keep_classes)

        raw_targets = getattr(base, "targets", None)
        if raw_targets is None:
            raise ValueError("ClassTruncatedDataset requires the base dataset to expose .targets")
        if isinstance(raw_targets, torch.Tensor):
            raw_targets = raw_targets.detach().cpu().tolist()

        target_transform = getattr(base, "target_transform", None)
        self.indices: list[int] = []
        for i, y in enumerate(raw_targets):
            yy = int(y)
            if target_transform is not None:
                yy = int(target_transform(yy))
            if yy in keep_set:
                self.indices.append(i)

        if not self.indices:
            raise ValueError(f"Class truncation produced an empty dataset for keep_classes={self.keep_classes}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[self.indices[idx]]
        y = int(y)
        return x, self.label_map[y]


def selected_classes_for_config(config: ExperimentConfig, base_num_classes: int) -> list[int]:
    if config.class_count is None:
        return list(range(base_num_classes))

    k = int(config.class_count)
    if k < 2:
        raise ValueError(f"--class-count must be >= 2, got {k}")
    if k > base_num_classes:
        raise ValueError(
            f"--class-count={k} exceeds available classes={base_num_classes} "
            f"for dataset={config.dataset!r}, emnist_split={config.emnist_split!r}"
        )

    selection = str(config.class_selection).lower()
    if selection == "first":
        return list(range(k))

    raise ValueError(
        f"Unsupported class_selection={config.class_selection!r}; currently only 'first' is supported."
    )

def make_dataloaders(config: ExperimentConfig, device: torch.device) -> tuple[DataLoader, DataLoader, int, int, int]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = config.dataset.lower()
    if dataset == "mnist":
        train_ds = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)
    elif dataset == "emnist":
        target_transform = emnist_letters_target_transform if config.emnist_split == "letters" else None
        train_ds = datasets.EMNIST(
            config.data_dir,
            split=config.emnist_split,
            train=True,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )
        test_ds = datasets.EMNIST(
            config.data_dir,
            split=config.emnist_split,
            train=False,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset!r}")

    base_num_classes = dataset_num_classes(dataset, config.emnist_split)
    selected_classes = selected_classes_for_config(config, base_num_classes)

    if config.class_count is not None:
        train_ds = ClassTruncatedDataset(train_ds, selected_classes)
        test_ds = ClassTruncatedDataset(test_ds, selected_classes)

    num_classes = len(selected_classes)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(derived_seed(config.seed, "dataloader_shuffle"))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data_bsize,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.data_bsize,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, test_loader, len(train_ds), len(test_ds), num_classes


def fade(t: torch.Tensor) -> torch.Tensor:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a + t * (b - a)


def sample_perlin_noise(batch_size: int, device: torch.device, height: int = 28, width: int = 28, res: int = 4) -> torch.Tensor:
    if res <= 0:
        raise ValueError(f"perlin_res must be positive, got {res}")
    ys = torch.arange(height, device=device, dtype=torch.float32) * (res / height)
    xs = torch.arange(width, device=device, dtype=torch.float32) * (res / width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    y0 = torch.floor(yy).long()
    x0 = torch.floor(xx).long()
    y1 = y0 + 1
    x1 = x0 + 1
    yf = yy - y0.float()
    xf = xx - x0.float()

    angles = 2 * torch.pi * torch.rand(batch_size, res + 1, res + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    g00 = gradients[:, y0, x0, :]
    g10 = gradients[:, y0, x1, :]
    g01 = gradients[:, y1, x0, :]
    g11 = gradients[:, y1, x1, :]

    d00 = torch.stack((xf, yf), dim=-1).unsqueeze(0)
    d10 = torch.stack((xf - 1.0, yf), dim=-1).unsqueeze(0)
    d01 = torch.stack((xf, yf - 1.0), dim=-1).unsqueeze(0)
    d11 = torch.stack((xf - 1.0, yf - 1.0), dim=-1).unsqueeze(0)

    n00 = (g00 * d00).sum(dim=-1)
    n10 = (g10 * d10).sum(dim=-1)
    n01 = (g01 * d01).sum(dim=-1)
    n11 = (g11 * d11).sum(dim=-1)

    u = fade(xf).unsqueeze(0)
    v = fade(yf).unsqueeze(0)
    nx0 = lerp(n00, n10, u)
    nx1 = lerp(n01, n11, u)
    noise = lerp(nx0, nx1, v).unsqueeze(1)
    return noise / (noise.abs().amax(dim=(1, 2, 3), keepdim=True) + EPS)


def sample_noise(
    batch_size: int,
    device: torch.device,
    noise_dist: str = "uniform",
    normalize: bool = False,
    perlin_res: int = 4,
) -> torch.Tensor:
    dist = noise_dist.lower()
    if dist == "uniform":
        x = 2.0 * torch.rand(batch_size, 1, 28, 28, device=device) - 1.0
    elif dist == "normal":
        x = torch.randn(batch_size, 1, 28, 28, device=device)
    elif dist == "perlin":
        x = sample_perlin_noise(batch_size, device=device, height=28, width=28, res=perlin_res)
    else:
        raise ValueError(f"Unknown noise_dist: {noise_dist!r}")
    if normalize:
        x = (x - 0.1307) / 0.3081
    return x


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

def set_trainable_layers(model: nn.Module, trainable_config: dict[str, bool]) -> None:
    for layer_name, is_trainable in trainable_config.items():
        layer = getattr(model, layer_name)
        for p in layer.parameters(recurse=False):
            p.requires_grad_(bool(is_trainable))


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_layer_parameters(model: nn.Module, layer_name: str) -> int:
    if not hasattr(model, layer_name):
        return 0
    return sum(p.numel() for p in getattr(model, layer_name).parameters(recurse=False))


def train_teacher(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    trainable_config: dict[str, bool],
    epochs: int,
    lr: float,
    seed: int,
) -> float:
    set_seed(seed)
    set_trainable_layers(model, trainable_config)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable teacher parameters.")
    model.train()
    opt = optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    last_epoch_loss = float("nan")
    for ep in range(epochs):
        running = 0.0
        batches = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, _ = model.forward_split(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item())
            batches += 1
        last_epoch_loss = running / max(1, batches)
        print(f"[Teacher] epoch {ep + 1:03d}/{epochs:03d} | class CE = {last_epoch_loss:.6f}")
    return last_epoch_loss


def distill_student_mse(
    student: nn.Module,
    teacher: nn.Module,
    device: torch.device,
    trainable_config: dict[str, bool],
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    noise_dist: str,
    perlin_res: int,
    normalize_noise: bool,
    seed: int,
) -> float:
    set_seed(seed)
    set_trainable_layers(student, trainable_config)
    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable student parameters.")
    student.train()
    teacher.eval()
    opt = optim.Adam(params, lr=lr)
    last_epoch_loss = float("nan")
    for ep in range(epochs):
        running = 0.0
        for _ in range(steps_per_epoch):
            x = sample_noise(
                batch_size=batch_size,
                device=device,
                noise_dist=noise_dist,
                normalize=normalize_noise,
                perlin_res=perlin_res,
            )
            with torch.no_grad():
                _, t_aux = teacher.forward_split(x)
            _, s_aux = student.forward_split(x)
            loss = F.mse_loss(s_aux, t_aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
        last_epoch_loss = running / max(1, steps_per_epoch)
        print(f"[Student] epoch {ep + 1:03d}/{epochs:03d} | aux MSE = {last_epoch_loss:.8f}")
    return last_epoch_loss


@torch.no_grad()
def evaluate_class(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model.forward_split(x)
        loss_sum += float(loss_fn(logits, y).item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    return correct / max(1, total), loss_sum / max(1, total)


@torch.no_grad()
def evaluate_aux_mse_on_noise(
    student: nn.Module,
    teacher: nn.Module,
    device: torch.device,
    batches: int,
    batch_size: int,
    noise_dist: str,
    perlin_res: int,
    normalize_noise: bool,
    seed: int,
) -> float:
    student.eval()
    teacher.eval()
    losses: list[float] = []
    with fork_torch_rng(device):
        set_seed(seed)
        for _ in range(batches):
            x = sample_noise(
                batch_size=batch_size,
                device=device,
                noise_dist=noise_dist,
                normalize=normalize_noise,
                perlin_res=perlin_res,
            )
            _, t_aux = teacher.forward_split(x)
            _, s_aux = student.forward_split(x)
            losses.append(float(F.mse_loss(s_aux, t_aux).item()))
    return float(np.mean(losses)) if losses else float("nan")


# -----------------------------------------------------------------------------
# Perturbations
# -----------------------------------------------------------------------------

def normalize_layer_selector(layers: Any, available_layers: list[str]) -> list[str]:
    if layers is None or layers == "all":
        return list(available_layers)
    if isinstance(layers, str):
        raw = layers.replace("+", ",").replace("|", ",")
        out = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        out = [str(x).strip() for x in layers if str(x).strip()]
    unknown = [x for x in out if x not in available_layers]
    if unknown:
        raise ValueError(f"Unknown perturb layers {unknown}; available layers are {available_layers}")
    return out


def parse_perturb_shorthand(value: str) -> dict[str, Any]:
    """Parse strings like 'student:aux_head,std=0.01,timing=before_student_training'."""
    text = value.strip()
    if not text:
        raise ValueError("Empty perturb shorthand")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    first = parts[0]
    if ":" not in first:
        raise ValueError(
            "Perturb shorthand must start with target:layers, e.g. "
            "student:aux_head,std=0.01,timing=before_student_training"
        )
    target, layers = [p.strip() for p in first.split(":", 1)]
    spec: dict[str, Any] = {"target": target, "layers": layers}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"Malformed perturb part {part!r}; expected key=value")
        key, val = [p.strip() for p in part.split("=", 1)]
        if key in {"std", "sigma"}:
            spec["std"] = float(val)
        elif key == "timing":
            spec["timing"] = val
        elif key == "include_weight":
            spec["include_weight"] = parse_bool_value(val)
        elif key == "include_bias":
            spec["include_bias"] = parse_bool_value(val)
        elif key == "distribution":
            spec["distribution"] = val
        else:
            spec[key] = val
    return spec


def canonicalize_perturb_specs(
    raw_specs: list[dict[str, Any]] | None,
    teacher_layers: list[str],
    student_layers: list[str],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_specs or []):
        spec = dict(raw)
        target = str(spec.get("target", "student")).lower()
        if target not in {"teacher", "student"}:
            raise ValueError(f"Perturb target must be teacher or student, got {target!r}")
        timing = str(spec.get("timing", "before_student_training"))
        if timing not in ALLOWED_TIMINGS:
            raise ValueError(f"Unsupported perturb timing {timing!r}; allowed={sorted(ALLOWED_TIMINGS)}")
        std = float(spec.get("std", spec.get("sigma", 0.0)))
        if std < 0:
            raise ValueError(f"Perturb std must be non-negative, got {std}")
        available = teacher_layers if target == "teacher" else student_layers
        layers = normalize_layer_selector(spec.get("layers", "all"), available)
        specs.append({
            "index": i,
            "target": target,
            "timing": timing,
            "layers": layers,
            "std": std,
            "include_weight": parse_bool_value(spec.get("include_weight", True)),
            "include_bias": parse_bool_value(spec.get("include_bias", True)),
            "distribution": str(spec.get("distribution", "normal")).lower(),
        })
    return specs


def tensor_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().reshape(-1).float(), ord=2).item())


def cosine_or_nan(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().reshape(-1).float()
    y = b.detach().reshape(-1).float()
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    if tensor_norm(x) < EPS or tensor_norm(y) < EPS:
        return float("nan")
    return float(F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=1).item())


@torch.no_grad()
def perturb_layer(
    layer: nn.Module,
    std: float,
    include_weight: bool,
    include_bias: bool,
    distribution: str,
) -> dict[str, float]:
    before_w = layer.weight.detach().clone() if hasattr(layer, "weight") and layer.weight is not None else None
    before_b = layer.bias.detach().clone() if hasattr(layer, "bias") and layer.bias is not None else None

    if include_weight and before_w is not None and std > 0:
        if distribution == "normal":
            layer.weight.add_(torch.randn_like(layer.weight) * std)
        else:
            raise ValueError(f"Unsupported perturb distribution: {distribution!r}")
    if include_bias and before_b is not None and std > 0:
        if distribution == "normal":
            layer.bias.add_(torch.randn_like(layer.bias) * std)
        else:
            raise ValueError(f"Unsupported perturb distribution: {distribution!r}")

    out: dict[str, float] = {}
    if before_w is not None:
        after_w = layer.weight.detach()
        delta_w = after_w - before_w
        out["weight_delta_norm"] = tensor_norm(delta_w)
        out["weight_before_norm"] = tensor_norm(before_w)
        out["weight_after_norm"] = tensor_norm(after_w)
        out["weight_delta_rel"] = out["weight_delta_norm"] / (out["weight_before_norm"] + EPS)
        out["weight_before_after_cosine"] = cosine_or_nan(before_w, after_w)
    else:
        out.update({
            "weight_delta_norm": float("nan"),
            "weight_before_norm": float("nan"),
            "weight_after_norm": float("nan"),
            "weight_delta_rel": float("nan"),
            "weight_before_after_cosine": float("nan"),
        })

    if before_b is not None:
        after_b = layer.bias.detach()
        delta_b = after_b - before_b
        out["bias_delta_norm"] = tensor_norm(delta_b)
        out["bias_before_norm"] = tensor_norm(before_b)
        out["bias_after_norm"] = tensor_norm(after_b)
        out["bias_delta_rel"] = out["bias_delta_norm"] / (out["bias_before_norm"] + EPS)
        out["bias_before_after_cosine"] = cosine_or_nan(before_b, after_b)
    else:
        out.update({
            "bias_delta_norm": float("nan"),
            "bias_before_norm": float("nan"),
            "bias_after_norm": float("nan"),
            "bias_delta_rel": float("nan"),
            "bias_before_after_cosine": float("nan"),
        })
    return out


def apply_perturbations_at_timing(
    timing: str,
    specs: list[dict[str, Any]],
    teacher: ModelBundle,
    student: ModelBundle,
    config: ExperimentConfig,
    run_id: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec["timing"] != timing:
            continue
        bundle = teacher if spec["target"] == "teacher" else student
        for layer_name in spec["layers"]:
            perturb_seed = derived_seed(config.seed, "perturb", spec["index"], timing, spec["target"], layer_name)
            with fork_torch_rng(device):
                set_seed(perturb_seed)
                metrics = perturb_layer(
                    getattr(bundle.model, layer_name),
                    std=spec["std"],
                    include_weight=spec["include_weight"],
                    include_bias=spec["include_bias"],
                    distribution=spec["distribution"],
                )
            row = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "seed": config.seed,
                "sweep_name": config.sweep_name,
                "run_label": config.run_label,
                "timing": timing,
                "target": spec["target"],
                "layer_name": layer_name,
                "std": spec["std"],
                "include_weight": int(spec["include_weight"]),
                "include_bias": int(spec["include_bias"]),
                "distribution": spec["distribution"],
                "perturb_seed": perturb_seed,
            }
            row.update(metrics)
            rows.append(row)
            print(
                f"[Perturb] {timing} | {spec['target']}.{layer_name} | "
                f"std={spec['std']} | w_rel={row['weight_delta_rel']:.6g} | b_rel={row['bias_delta_rel']:.6g}"
            )
    return rows


# -----------------------------------------------------------------------------
# Snapshots and layer metrics
# -----------------------------------------------------------------------------

def snapshot_model(model: nn.Module, layer_names: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    snap: dict[str, dict[str, torch.Tensor]] = {}
    for name in layer_names:
        layer = getattr(model, name)
        entry: dict[str, torch.Tensor] = {}
        if hasattr(layer, "weight") and layer.weight is not None:
            entry["weight"] = layer.weight.detach().cpu().clone()
        if hasattr(layer, "bias") and layer.bias is not None:
            entry["bias"] = layer.bias.detach().cpu().clone()
        snap[name] = entry
    return snap


def shape_string(x: torch.Tensor | None) -> str:
    if x is None:
        return ""
    return "x".join(str(v) for v in tuple(x.shape))


def compare_snapshot_tensors(
    run_id: str,
    config: ExperimentConfig,
    comparison: str,
    reference_model: str,
    other_model: str,
    reference_state: str,
    other_state: str,
    reference_snapshot: dict[str, dict[str, torch.Tensor]],
    other_snapshot: dict[str, dict[str, torch.Tensor]],
    layer_order: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_name in layer_order:
        ref_layer = reference_snapshot.get(layer_name)
        oth_layer = other_snapshot.get(layer_name)
        layer_status = "comparable"
        if ref_layer is None and oth_layer is None:
            layer_status = "missing_both"
        elif ref_layer is None:
            layer_status = f"missing_in_{reference_model}"
        elif oth_layer is None:
            layer_status = f"missing_in_{other_model}"

        for tensor_kind in ["weight", "bias"]:
            ref = ref_layer.get(tensor_kind) if ref_layer is not None else None
            oth = oth_layer.get(tensor_kind) if oth_layer is not None else None
            status = layer_status
            cosine = diff_norm = ref_norm = other_norm = rel_diff_ref = rel_diff_sym = float("nan")
            if status == "comparable":
                if ref is None and oth is None:
                    status = "missing_tensor_both"
                elif ref is None:
                    status = f"missing_tensor_in_{reference_model}"
                elif oth is None:
                    status = f"missing_tensor_in_{other_model}"
                elif tuple(ref.shape) != tuple(oth.shape):
                    status = "shape_mismatch"
                else:
                    ref_norm = tensor_norm(ref)
                    other_norm = tensor_norm(oth)
                    diff_norm = tensor_norm(ref - oth)
                    cosine = cosine_or_nan(ref, oth)
                    rel_diff_ref = diff_norm / (ref_norm + EPS)
                    rel_diff_sym = 2.0 * diff_norm / (ref_norm + other_norm + EPS)
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "seed": config.seed,
                "sweep_name": config.sweep_name,
                "run_label": config.run_label,
                "comparison": comparison,
                "reference_model": reference_model,
                "other_model": other_model,
                "reference_state": reference_state,
                "other_state": other_state,
                "layer_name": layer_name,
                "tensor_kind": tensor_kind,
                "reference_shape": shape_string(ref),
                "other_shape": shape_string(oth),
                "status": status,
                "cosine": cosine,
                "diff_norm": diff_norm,
                "reference_norm": ref_norm,
                "other_norm": other_norm,
                "relative_diff_reference": rel_diff_ref,
                "relative_diff_symmetric": rel_diff_sym,
            })
    return rows


def make_all_layer_metrics(
    run_id: str,
    config: ExperimentConfig,
    teacher_layers: list[str],
    student_layers: list[str],
    teacher_initial: dict[str, dict[str, torch.Tensor]],
    teacher_final: dict[str, dict[str, torch.Tensor]],
    student_initial: dict[str, dict[str, torch.Tensor]],
    student_final: dict[str, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    layer_order = ordered_unique(teacher_layers + student_layers)
    rows: list[dict[str, Any]] = []
    rows += compare_snapshot_tensors(
        run_id, config,
        comparison="teacher_student_init",
        reference_model="teacher",
        other_model="student",
        reference_state="initial",
        other_state="initial",
        reference_snapshot=teacher_initial,
        other_snapshot=student_initial,
        layer_order=layer_order,
    )
    rows += compare_snapshot_tensors(
        run_id, config,
        comparison="teacher_student_final",
        reference_model="teacher",
        other_model="student",
        reference_state="final",
        other_state="final",
        reference_snapshot=teacher_final,
        other_snapshot=student_final,
        layer_order=layer_order,
    )
    rows += compare_snapshot_tensors(
        run_id, config,
        comparison="teacher_init_final",
        reference_model="teacher",
        other_model="teacher",
        reference_state="initial",
        other_state="final",
        reference_snapshot=teacher_initial,
        other_snapshot=teacher_final,
        layer_order=teacher_layers,
    )
    rows += compare_snapshot_tensors(
        run_id, config,
        comparison="student_init_final",
        reference_model="student",
        other_model="student",
        reference_state="initial",
        other_state="final",
        reference_snapshot=student_initial,
        other_snapshot=student_final,
        layer_order=student_layers,
    )
    return rows


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def seed_output_dir(outdir: str, seed: int) -> Path:
    return Path(outdir) / f"seed_{int(seed):06d}"


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_unique(k for row in rows for k in row.keys())
    file_exists = path.exists() and path.stat().st_size > 0
    if file_exists:
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            old_header = next(reader)
        fieldnames = ordered_unique(old_header + fieldnames)
        if fieldnames != old_header:
            # Rewrite with expanded schema. This is safe for per-seed sequential appends.
            existing: list[dict[str, Any]] = []
            with path.open("r", newline="") as f:
                r = csv.DictReader(f)
                existing.extend(dict(row) for row in r)
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in existing:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
                for row in rows:
                    w.writerow({k: csv_value(row.get(k, "")) for k in fieldnames})
            return
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for row in rows:
            w.writerow({k: csv_value(row.get(k, "")) for k in fieldnames})


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    if isinstance(value, bool):
        return int(value)
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def output_paths(config: ExperimentConfig) -> dict[str, Path]:
    sdir = seed_output_dir(config.outdir, config.seed)
    paths = {
        "seed_dir": sdir,
        "runs": sdir / "runs.csv",
        "layers": sdir / "layer_metrics.csv",
        "perturbations": sdir / "perturbations.csv",
        "configs": sdir / "configs.jsonl",
    }
    if config.append_global:
        paths.update({
            "global_runs": Path(config.outdir) / "runs_all_seeds.csv",
            "global_layers": Path(config.outdir) / "layer_metrics_all_seeds.csv",
            "global_perturbations": Path(config.outdir) / "perturbations_all_seeds.csv",
        })
    return paths


# -----------------------------------------------------------------------------
# Main experiment runner
# -----------------------------------------------------------------------------

def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def flatten_config_for_run(config: ExperimentConfig) -> dict[str, Any]:
    d = dict(config.__dict__)
    d["teacher_hidden_dims"] = list(config.teacher_hidden_dims)
    d["student_hidden_dims"] = list(config.student_hidden_dims)
    return d


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    t0 = time.time()
    run_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now().isoformat(timespec="seconds")
    device = select_device(config.device)
    set_seed(derived_seed(config.seed, "experiment_start"))

    train_loader, test_loader, train_size, test_size, num_classes = make_dataloaders(config, device)

    teacher = build_model_bundle(
        model_name="teacher",
        model_type=config.teacher_type,
        hidden_dims=config.teacher_hidden_dims,
        arch_spec=config.teacher_arch_spec,
        init_config_raw=config.teacher_init,
        trainable_config_raw=config.teacher_trainable,
        m=config.m,
        num_classes=num_classes,
        seed=config.seed,
        device=device,
    )
    student = build_model_bundle(
        model_name="student",
        model_type=config.student_type,
        hidden_dims=config.student_hidden_dims,
        arch_spec=config.student_arch_spec,
        init_config_raw=config.student_init,
        trainable_config_raw=config.student_trainable,
        m=config.m,
        num_classes=num_classes,
        seed=config.seed,
        device=device,
    )

    perturb_specs = canonicalize_perturb_specs(
        config.perturb_specs,
        teacher.layer_names,
        student.layer_names,
    )

    perturb_rows: list[dict[str, Any]] = []
    perturb_rows += apply_perturbations_at_timing(
        "before_teacher_training", perturb_specs, teacher, student, config, run_id, device
    )

    teacher_initial = snapshot_model(teacher.model, teacher.layer_names)
    teacher_init_acc, teacher_init_class_loss = evaluate_class(teacher.model, test_loader, device)
    print(f"[Eval] teacher initial | acc={teacher_init_acc:.6f} | class CE={teacher_init_class_loss:.6f}")

    teacher_train_loss_last = train_teacher(
        teacher.model,
        train_loader,
        device,
        trainable_config=teacher.trainable_config,
        epochs=config.teacher_epochs,
        lr=config.teacher_lr,
        seed=derived_seed(config.seed, "teacher_training"),
    )

    perturb_rows += apply_perturbations_at_timing(
        "after_teacher_training", perturb_specs, teacher, student, config, run_id, device
    )
    teacher_final = snapshot_model(teacher.model, teacher.layer_names)
    teacher_final_acc, teacher_final_class_loss = evaluate_class(teacher.model, test_loader, device)
    print(f"[Eval] teacher final   | acc={teacher_final_acc:.6f} | class CE={teacher_final_class_loss:.6f}")

    perturb_rows += apply_perturbations_at_timing(
        "before_student_training", perturb_specs, teacher, student, config, run_id, device
    )
    student_initial = snapshot_model(student.model, student.layer_names)
    student_init_acc, student_init_class_loss = evaluate_class(student.model, test_loader, device)
    student_init_aux_loss = evaluate_aux_mse_on_noise(
        student.model,
        teacher.model,
        device,
        batches=config.eval_noise_batches,
        batch_size=config.eval_noise_bsize,
        noise_dist=config.noise_dist,
        perlin_res=config.perlin_res,
        normalize_noise=config.normalize_noise,
        seed=derived_seed(config.seed, "eval_noise", "student_initial"),
    )
    print(
        f"[Eval] student initial | acc={student_init_acc:.6f} | "
        f"class CE={student_init_class_loss:.6f} | aux MSE={student_init_aux_loss:.8f}"
    )

    student_train_aux_loss_last = distill_student_mse(
        student.model,
        teacher.model,
        device,
        trainable_config=student.trainable_config,
        epochs=config.student_epochs,
        steps_per_epoch=config.noise_steps,
        batch_size=config.noise_bsize,
        lr=config.student_lr,
        noise_dist=config.noise_dist,
        perlin_res=config.perlin_res,
        normalize_noise=config.normalize_noise,
        seed=derived_seed(config.seed, "student_training_noise"),
    )

    perturb_rows += apply_perturbations_at_timing(
        "after_student_training", perturb_specs, teacher, student, config, run_id, device
    )
    student_final = snapshot_model(student.model, student.layer_names)
    student_final_acc, student_final_class_loss = evaluate_class(student.model, test_loader, device)
    student_final_aux_loss = evaluate_aux_mse_on_noise(
        student.model,
        teacher.model,
        device,
        batches=config.eval_noise_batches,
        batch_size=config.eval_noise_bsize,
        noise_dist=config.noise_dist,
        perlin_res=config.perlin_res,
        normalize_noise=config.normalize_noise,
        seed=derived_seed(config.seed, "eval_noise", "student_final"),
    )
    print(
        f"[Eval] student final   | acc={student_final_acc:.6f} | "
        f"class CE={student_final_class_loss:.6f} | aux MSE={student_final_aux_loss:.8f}"
    )

    layer_rows = make_all_layer_metrics(
        run_id=run_id,
        config=config,
        teacher_layers=teacher.layer_names,
        student_layers=student.layer_names,
        teacher_initial=teacher_initial,
        teacher_final=teacher_final,
        student_initial=student_initial,
        student_final=student_final,
    )

    runtime_seconds = time.time() - t0
    seed_info = {
        "rng_base": rng_base(config.seed),
        "teacher_model_seed": derived_seed(config.seed, "teacher", "model_random_init"),
        "student_model_seed": derived_seed(config.seed, "student", "model_random_init"),
        "teacher_training_seed": derived_seed(config.seed, "teacher_training"),
        "student_training_seed": derived_seed(config.seed, "student_training_noise"),
        "dataloader_seed": derived_seed(config.seed, "dataloader_shuffle"),
    }

    run_row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": timestamp,
        "runtime_seconds": runtime_seconds,
        "sweep_name": config.sweep_name,
        "run_label": config.run_label,
        "seed": config.seed,
        **seed_info,
        "torch_version": torch.__version__,
        "cuda_available": int(torch.cuda.is_available()),
        "device": str(device),
        "dataset": config.dataset,
        "emnist_split": config.emnist_split,
        "train_size": train_size,
        "test_size": test_size,
        "num_classes": num_classes,
        "data_dir": config.data_dir,
        "outdir": config.outdir,
        "teacher_type": config.teacher_type,
        "student_type": config.student_type,
        "teacher_hidden_dims": config.teacher_hidden_dims,
        "student_hidden_dims": config.student_hidden_dims,
        "teacher_arch_description": teacher.arch_description,
        "student_arch_description": student.arch_description,
        "teacher_layer_names": teacher.layer_names,
        "student_layer_names": student.layer_names,
        "m": config.m,
        "teacher_epochs": config.teacher_epochs,
        "student_epochs": config.student_epochs,
        "data_bsize": config.data_bsize,
        "noise_bsize": config.noise_bsize,
        "noise_steps": config.noise_steps,
        "teacher_lr": config.teacher_lr,
        "student_lr": config.student_lr,
        "noise_dist": config.noise_dist,
        "perlin_res": config.perlin_res,
        "normalize_noise": int(config.normalize_noise),
        "eval_noise_batches": config.eval_noise_batches,
        "eval_noise_bsize": config.eval_noise_bsize,
        "teacher_init_config": teacher.init_config,
        "student_init_config": student.init_config,
        "teacher_trainable_config": teacher.trainable_config,
        "student_trainable_config": student.trainable_config,
        "perturb_specs": perturb_specs,
        "teacher_total_parameters": count_total_parameters(teacher.model),
        "student_total_parameters": count_total_parameters(student.model),
        "teacher_trainable_parameters": count_trainable_parameters(teacher.model),
        "student_trainable_parameters": count_trainable_parameters(student.model),
        "teacher_init_acc": teacher_init_acc,
        "teacher_final_acc": teacher_final_acc,
        "student_init_acc": student_init_acc,
        "student_final_acc": student_final_acc,
        "teacher_init_class_loss": teacher_init_class_loss,
        "teacher_final_class_loss": teacher_final_class_loss,
        "student_init_class_loss": student_init_class_loss,
        "student_final_class_loss": student_final_class_loss,
        "student_init_aux_loss": student_init_aux_loss,
        "student_final_aux_loss": student_final_aux_loss,
        "teacher_train_loss_last": teacher_train_loss_last,
        "student_train_aux_loss_last": student_train_aux_loss_last,
        "class_count": config.class_count if config.class_count is not None else "",
        "class_selection": config.class_selection,
        "base_num_classes": dataset_num_classes(config.dataset, config.emnist_split),
    }

    # Add compact aliases for the most frequently inspected head metrics.
    run_row.update(metric_aliases_from_layer_rows(layer_rows))

    paths = output_paths(config)
    append_csv(paths["runs"], [run_row])
    append_csv(paths["layers"], layer_rows)
    append_csv(paths["perturbations"], perturb_rows)
    append_jsonl(paths["configs"], [{
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": timestamp,
        "effective_config": flatten_config_for_run(config),
        "teacher_init_config": teacher.init_config,
        "student_init_config": student.init_config,
        "teacher_trainable_config": teacher.trainable_config,
        "student_trainable_config": student.trainable_config,
        "teacher_arch_description": teacher.arch_description,
        "student_arch_description": student.arch_description,
        "seed_info": seed_info,
    }])

    if config.append_global:
        append_csv(paths["global_runs"], [run_row])
        append_csv(paths["global_layers"], layer_rows)
        append_csv(paths["global_perturbations"], perturb_rows)

    print(f"[Output] seed directory: {paths['seed_dir']}")
    print(f"[Output] run_id: {run_id}")
    return {
        "run_id": run_id,
        "paths": {k: str(v) for k, v in paths.items()},
        "run_row": run_row,
        "layer_rows": layer_rows,
        "perturbation_rows": perturb_rows,
    }


def metric_aliases_from_layer_rows(layer_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    wanted = {
        ("teacher_student_init", "class_head", "weight", "cosine"): "cos_init_teacher_student_class_head_weight",
        ("teacher_student_init", "class_head", "bias", "cosine"): "cos_init_teacher_student_class_head_bias",
        ("teacher_student_init", "aux_head", "weight", "cosine"): "cos_init_teacher_student_aux_head_weight",
        ("teacher_student_init", "aux_head", "bias", "cosine"): "cos_init_teacher_student_aux_head_bias",
        ("teacher_student_final", "class_head", "weight", "cosine"): "cos_final_teacher_student_class_head_weight",
        ("teacher_student_final", "class_head", "bias", "cosine"): "cos_final_teacher_student_class_head_bias",
        ("teacher_student_final", "aux_head", "weight", "cosine"): "cos_final_teacher_student_aux_head_weight",
        ("teacher_student_final", "aux_head", "bias", "cosine"): "cos_final_teacher_student_aux_head_bias",
        ("student_init_final", "class_head", "weight", "cosine"): "cos_student_init_final_class_head_weight",
        ("student_init_final", "class_head", "bias", "cosine"): "cos_student_init_final_class_head_bias",
        ("student_init_final", "aux_head", "weight", "cosine"): "cos_student_init_final_aux_head_weight",
        ("student_init_final", "aux_head", "bias", "cosine"): "cos_student_init_final_aux_head_bias",
        ("teacher_init_final", "class_head", "weight", "cosine"): "cos_teacher_init_final_class_head_weight",
        ("teacher_init_final", "class_head", "bias", "cosine"): "cos_teacher_init_final_class_head_bias",
        ("teacher_init_final", "aux_head", "weight", "cosine"): "cos_teacher_init_final_aux_head_weight",
        ("teacher_init_final", "aux_head", "bias", "cosine"): "cos_teacher_init_final_aux_head_bias",
    }
    # Same aliases for relative differences.
    for row in layer_rows:
        key = (row["comparison"], row["layer_name"], row["tensor_kind"], "cosine")
        if key in wanted:
            aliases[wanted[key]] = row["cosine"]
        rel_key = (row["comparison"], row["layer_name"], row["tensor_kind"], "relative")
        base = wanted.get((row["comparison"], row["layer_name"], row["tensor_kind"], "cosine"))
        if base:
            aliases[base.replace("cos_", "rel_fro_")] = row["relative_diff_reference"]
            aliases[base.replace("cos_", "diff_norm_")] = row["diff_norm"]
    return aliases
