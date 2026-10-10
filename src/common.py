"""Shared helpers: paths, config loading, seeding, JSONL logging, environment capture."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

PAD, UNK, EOS = 0, 1, 2  # special token ids (TASK §2)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply command-line overrides (only keys that are not None)."""
    with open(ROOT / path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def apply_limit(cfg: dict[str, Any]) -> dict[str, Any]:
    """Smoke mode: if `limit_reviews` is set, redirect outputs so the full data is never clobbered."""
    if cfg.get("limit_reviews"):
        cfg["processed_dir"] = "data/smoke/processed"
        cfg["streams_dir"] = "data/smoke/streams"
        cfg["results_dir"] = "results/smoke"
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is always installed in this project
        pass


def write_json(obj: Any, path: str | Path) -> None:
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=_json_default)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with open(ROOT / path) as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class JsonlLogger:
    """Append-only JSONL log, one record per line."""

    def __init__(self, path: str | Path):
        self.path = ROOT / path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def log(self, record: dict[str, Any]) -> None:
        self._f.write(json.dumps(record, default=_json_default) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(ROOT / path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001 - best effort only
        return None


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def env_info(with_gpu: bool = True) -> dict[str, Any]:
    """Hardware and library versions (TASK §4.4); also a load/GPU snapshot for shared-node context."""
    import torch

    info: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_model": cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "loadavg_1_5_15": os.getloadavg(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    for mod in ("pandas", "sklearn", "gensim", "datasets", "matplotlib", "yaml", "scipy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    if with_gpu and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)
        free, total = torch.cuda.mem_get_info(0)
        info["gpu_mem_free_gib"] = round(free / 2**30, 2)
        info["gpu_mem_total_gib"] = round(total / 2**30, 2)
        smi = _run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader"])
        info["nvidia_smi_snapshot"] = smi.splitlines() if smi else None
        info["nvidia_driver"] = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    else:
        info["gpu_name"] = None
    return info


def sha256_array(x) -> str:
    """Checksum of a tensor/array's bytes (used to verify frozen embedding tables)."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous().numpy()
    except ImportError:  # pragma: no cover
        pass
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def fmt_seconds(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def smoke_indices(labels, n: int | None):
    """Smoke mode: indices of the first n//2 reviews of each class (HF IMDb splits are sorted by label);
    identity (all indices) when n is None."""
    labels = np.asarray(labels)
    if not n:
        return np.arange(len(labels))
    neg = np.flatnonzero(labels == 0)[: n // 2]
    pos = np.flatnonzero(labels == 1)[: n - n // 2]
    return np.sort(np.concatenate([neg, pos]))
