#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
MODEL_ID = os.environ.get("QWEN_RT_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")

BASE_PACKAGES = [
    "numpy==2.4.2",
    "av==16.1.0",
    "fastapi==0.135.1",
    "uvicorn[standard]==0.41.0",
    "websockets==16.0",
    "huggingface_hub==0.36.2",
    "transformers==4.57.6",
    "python-multipart==0.0.20",
    "qwen-asr==0.0.6",
]


@dataclass(slots=True)
class GpuInfo:
    name: str
    compute_cap: tuple[int, int]
    memory_mib: int
    driver_version: str


@dataclass(slots=True)
class TorchProfile:
    name: str
    torch: str
    torchvision: str
    torchaudio: str
    index_url: str
    min_driver_major: int | None = None


TORCH_PROFILES = [
    TorchProfile(
        name="cu128",
        torch="2.7.1",
        torchvision="0.22.1",
        torchaudio="2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
        min_driver_major=570,
    ),
    TorchProfile(
        name="cu124",
        torch="2.6.0",
        torchvision="0.21.0",
        torchaudio="2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
        min_driver_major=550,
    ),
    TorchProfile(
        name="cu121",
        torch="2.5.1",
        torchvision="0.20.1",
        torchaudio="2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
        min_driver_major=535,
    ),
    TorchProfile(
        name="cpu",
        torch="2.6.0",
        torchvision="0.21.0",
        torchaudio="2.6.0",
        index_url="https://download.pytorch.org/whl/cpu",
        min_driver_major=None,
    ),
]


def run(cmd: list[str], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def detect_gpu() -> GpuInfo | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None

    line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not line:
        return None

    name, compute_cap, memory_total, driver_version = [part.strip() for part in line.split(",", 3)]
    major, minor = compute_cap.split(".", 1)
    memory_mib = int(memory_total.split()[0])
    return GpuInfo(
        name=name,
        compute_cap=(int(major), int(minor)),
        memory_mib=memory_mib,
        driver_version=driver_version,
    )


def choose_torch_profile(gpu: GpuInfo | None) -> TorchProfile:
    if gpu is None:
        return next(profile for profile in TORCH_PROFILES if profile.name == "cpu")

    cc_major, _ = gpu.compute_cap
    driver_major = int(gpu.driver_version.split(".", 1)[0])
    if cc_major >= 12:
        for profile in TORCH_PROFILES:
            if profile.name == "cu128" and driver_major >= (profile.min_driver_major or 0):
                return profile
        raise SystemExit(
            f"GPU compute capability {cc_major}.x requires a newer PyTorch CUDA wheel path. "
            f"Current driver {gpu.driver_version} is too old for the configured cu128 profile."
        )

    for profile in TORCH_PROFILES:
        if profile.min_driver_major is None:
            continue
        if profile.name == "cu128":
            continue
        if driver_major >= profile.min_driver_major:
            return profile
    raise SystemExit(
        f"Unsupported NVIDIA driver {gpu.driver_version}. "
        "Need >= 535 for a supported CUDA wheel path."
    )


def repo_cache_dir(model_id: str) -> Path:
    return ROOT / ".cache" / "hf" / "hub" / f"models--{model_id.replace('/', '--')}"


def resolve_snapshot_dir(model_id: str) -> Path | None:
    snapshots_dir = repo_cache_dir(model_id) / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def build_runtime_env(gpu: GpuInfo | None) -> dict[str, str]:
    env = {
        "HF_HOME": str(ROOT / ".cache" / "hf"),
        "HUGGINGFACE_HUB_CACHE": str(ROOT / ".cache" / "hf" / "hub"),
        "TRANSFORMERS_CACHE": str(ROOT / ".cache" / "hf" / "transformers"),
        "XDG_CACHE_HOME": str(ROOT / ".cache" / "xdg"),
        "QWEN_RT_MODEL_ID": MODEL_ID,
        "QWEN_RT_MODEL_NAME": "qwen3-asr-rt",
        "QWEN_RT_LOCAL_FILES_ONLY": "false",
    }
    if gpu is None:
        env.update(
            {
                "QWEN_RT_DEVICE_MAP": "cpu",
                "QWEN_RT_MODEL_DTYPE": "float32",
                "QWEN_RT_ATTN_IMPLEMENTATION": "eager",
                "QWEN_RT_MAX_INFERENCE_BATCH_SIZE": "1",
            }
        )
        return env

    cc_major, _ = gpu.compute_cap
    memory_gib = gpu.memory_mib / 1024.0
    if cc_major >= 8:
        env.update(
            {
                "QWEN_RT_DEVICE_MAP": "cuda:0",
                "QWEN_RT_MODEL_DTYPE": "bfloat16",
                "QWEN_RT_ATTN_IMPLEMENTATION": "sdpa",
                "QWEN_RT_MAX_INFERENCE_BATCH_SIZE": "8" if memory_gib >= 16 else "4",
            }
        )
    elif cc_major >= 7:
        env.update(
            {
                "QWEN_RT_DEVICE_MAP": "cuda:0",
                "QWEN_RT_MODEL_DTYPE": "float16",
                "QWEN_RT_ATTN_IMPLEMENTATION": "sdpa",
                "QWEN_RT_MAX_INFERENCE_BATCH_SIZE": "6" if memory_gib >= 16 else "3",
            }
        )
    elif cc_major >= 6:
        env.update(
            {
                "QWEN_RT_DEVICE_MAP": "cuda:0",
                "QWEN_RT_MODEL_DTYPE": "float16",
                "QWEN_RT_ATTN_IMPLEMENTATION": "eager",
                "QWEN_RT_MAX_INFERENCE_BATCH_SIZE": "4" if memory_gib >= 16 else "2",
            }
        )
    else:
        raise SystemExit(
            f"Unsupported GPU compute capability {gpu.compute_cap[0]}.{gpu.compute_cap[1]}. "
            "Qwen3-ASR GPU mode needs at least SM 6.0."
        )
    return env


def write_runtime_env(env: dict[str, str], dry_run: bool = False) -> None:
    env_path = ROOT / ".env.runtime"
    lines = [f'export {key}="{value}"' for key, value in sorted(env.items())]
    text = "\n".join(lines) + "\n"
    print(f"+ write {env_path}")
    if not dry_run:
        env_path.write_text(text, encoding="utf-8")


def ensure_venv(dry_run: bool = False) -> None:
    if VENV_PYTHON.exists():
        return
    run([sys.executable, "-m", "venv", str(VENV_DIR)], dry_run=dry_run)


def pip_install(packages: list[str], index_url: str | None = None, dry_run: bool = False) -> None:
    cmd = [str(VENV_PYTHON), "-m", "pip", "install"]
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.extend(packages)
    run(cmd, dry_run=dry_run)


def maybe_download_model(runtime_env: dict[str, str], dry_run: bool = False) -> None:
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={MODEL_ID!r}, cache_dir={str(ROOT / '.cache' / 'hf')!r})"
    )
    env = os.environ.copy()
    env.update(runtime_env)
    run([str(VENV_PYTHON), "-c", code], env=env, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-model-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu = detect_gpu()
    torch_profile = choose_torch_profile(gpu)
    runtime_env = build_runtime_env(gpu)
    final_runtime_env = dict(runtime_env)
    cached_snapshot = resolve_snapshot_dir(MODEL_ID)
    if cached_snapshot is not None:
        final_runtime_env["QWEN_RT_MODEL_PATH"] = str(cached_snapshot)
        final_runtime_env["QWEN_RT_LOCAL_FILES_ONLY"] = "true"
        final_runtime_env["HF_HUB_OFFLINE"] = "1"
        final_runtime_env["TRANSFORMERS_OFFLINE"] = "1"
    elif not args.skip_model_download:
        final_runtime_env["QWEN_RT_LOCAL_FILES_ONLY"] = "true"
        final_runtime_env["HF_HUB_OFFLINE"] = "1"
        final_runtime_env["TRANSFORMERS_OFFLINE"] = "1"

    summary = {
        "gpu": None
        if gpu is None
        else {
            "name": gpu.name,
            "compute_cap": f"{gpu.compute_cap[0]}.{gpu.compute_cap[1]}",
            "memory_mib": gpu.memory_mib,
            "driver_version": gpu.driver_version,
        },
        "torch_profile": torch_profile.name,
        "runtime_env": final_runtime_env,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    ensure_venv(dry_run=args.dry_run)
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], dry_run=args.dry_run)
    pip_install(
        [
            f"torch=={torch_profile.torch}",
            f"torchvision=={torch_profile.torchvision}",
            f"torchaudio=={torch_profile.torchaudio}",
        ],
        index_url=torch_profile.index_url,
        dry_run=args.dry_run,
    )
    pip_install(BASE_PACKAGES, dry_run=args.dry_run)

    if not args.skip_model_download:
        maybe_download_model(runtime_env, dry_run=args.dry_run)
        cached_snapshot = resolve_snapshot_dir(MODEL_ID)
        if cached_snapshot is not None:
            final_runtime_env["QWEN_RT_MODEL_PATH"] = str(cached_snapshot)
        runtime_env = final_runtime_env
    elif cached_snapshot is not None:
        runtime_env = final_runtime_env

    write_runtime_env(runtime_env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
