"""
Single Modal app for the entire knowledge distillation pipeline.

All four phases run on Modal GPUs, sharing a persistent volume for
checkpoints and the cached CIFAR-100 dataset.

──────────────────────────────────────────────────────────────────────
USAGE — run these from your local machine in order:

  pip install modal
  modal setup                              # one-time auth

  modal run modal_app.py::train_teacher    # Phase 1: ~45 min on T4
  modal run modal_app.py::train_baseline   # Phase 2: ~20 min on T4
  modal run modal_app.py::distill          # Phase 3: ~25 min on T4
  modal run modal_app.py::compare          # Phase 4: ~30 sec

Or run the whole pipeline end-to-end with one command:

  modal run modal_app.py::run_all          # ~90 min total

For hyperparameter sweeps:

  modal run modal_app.py::distill --temperature 8.0 --alpha 0.9
  modal run modal_app.py::ablation         # 6 configs in parallel

To download a checkpoint locally:

  modal volume get kd-results /checkpoints/student_distilled.pth ./

To inspect what's on the volume:

  modal volume ls kd-results /checkpoints

──────────────────────────────────────────────────────────────────────
Volume layout:
  /data/                         CIFAR-100 cache (downloaded once)
  /checkpoints/teacher.pth       Phase 1 output
  /checkpoints/student_baseline.pth   Phase 2 output
  /checkpoints/student_distilled_<run_name>.pth  Phase 3 output(s)
  /logs/<run_name>.json          Per-epoch training metrics
"""

import modal

app = modal.App("kd-cifar100")

# ── Container image ──────────────────────────────────────────────────
# CUDA-enabled PyTorch + torchvision is all we need.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy<2.0",
    )
    # Ship our local source files into the image so the Modal functions
    # can import them.
    .add_local_python_source(
        "models", "losses", "data", "train_standard", "train_distill", "compare"
    )
)

# Persistent volume — survives across runs, shared between functions
volume = modal.Volume.from_name("kd-results", create_if_missing=True)

# Standard mount points used by every function
VOL_MOUNT = "/vol"
DATA_DIR = f"{VOL_MOUNT}/data"
CKPT_DIR = f"{VOL_MOUNT}/checkpoints"
LOG_DIR  = f"{VOL_MOUNT}/logs"

# GPU choice — T4 is cheapest and plenty for CIFAR ResNets.
# Bump to "A10G" for ~2× speedup at higher cost.
GPU_TYPE = "T4"


# ─────────────────────────────────────────────────────────────────────
# Helper: ensure directories exist on the volume
# ─────────────────────────────────────────────────────────────────────
def _setup_dirs():
    """Create the standard subdirectories on the volume if missing."""
    import os
    for d in (DATA_DIR, CKPT_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# PHASE 1: Train the teacher (ResNet-56)
# ─────────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600 * 2,
    volumes={VOL_MOUNT: volume},
)
def train_teacher(
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.1,
):
    """Phase 1: train ResNet-56 from scratch on CIFAR-100."""
    import sys
    import torch
    _setup_dirs()

    print("=" * 60)
    print("PHASE 1: TRAINING TEACHER (ResNet-56)")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_path = f"{CKPT_DIR}/teacher.pth"
    sys.argv = [
        "train_standard.py",
        "--model", "resnet56",
        "--out", out_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--data-root", DATA_DIR,
        "--num-workers", "2",
    ]
    import train_standard
    train_standard.main()

    volume.commit()
    print(f"\n[✓] Teacher saved to {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# PHASE 2: Train the baseline student (ResNet-20, no KD)
# ─────────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,
    volumes={VOL_MOUNT: volume},
)
def train_baseline(
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.1,
):
    """Phase 2: train ResNet-20 from scratch on CIFAR-100 (no distillation)."""
    import sys
    import torch
    _setup_dirs()

    print("=" * 60)
    print("PHASE 2: TRAINING BASELINE STUDENT (ResNet-20)")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_path = f"{CKPT_DIR}/student_baseline.pth"
    sys.argv = [
        "train_standard.py",
        "--model", "resnet20",
        "--out", out_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--data-root", DATA_DIR,
        "--num-workers", "2",
    ]
    import train_standard
    train_standard.main()

    volume.commit()
    print(f"\n[✓] Baseline student saved to {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# PHASE 3: Distill teacher → student
# ─────────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600 * 2,
    volumes={VOL_MOUNT: volume},
)
def distill(
    run_name: str = "default",
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.1,
    temperature: float = 4.0,
    alpha: float = 0.7,
    beta: float = 100.0,
):
    """Phase 3: distill the teacher into a fresh ResNet-20 student."""
    import os
    import sys
    import torch
    _setup_dirs()

    teacher_ckpt = f"{CKPT_DIR}/teacher.pth"
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {teacher_ckpt}.\n"
            f"Run `modal run modal_app.py::train_teacher` first."
        )

    print("=" * 60)
    print(f"PHASE 3: DISTILLATION (run: {run_name})")
    print("=" * 60)
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"T={temperature}  α={alpha}  β={beta}  epochs={epochs}")

    out_path = f"{CKPT_DIR}/student_distilled_{run_name}.pth"
    log_path = f"{LOG_DIR}/{run_name}.json"

    sys.argv = [
        "train_distill.py",
        "--teacher-ckpt", teacher_ckpt,
        "--out", out_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--temperature", str(temperature),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--data-root", DATA_DIR,
        "--num-workers", "2",
        "--log-json", log_path,
    ]
    import train_distill
    train_distill.main()

    volume.commit()
    print(f"\n[✓] Distilled student saved to {out_path}")
    print(f"[✓] Training log saved to {log_path}")
    return {"checkpoint": out_path, "log": log_path}


# ─────────────────────────────────────────────────────────────────────
# PHASE 4: Compare all three models
# ─────────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu=GPU_TYPE,            # GPU for accurate latency timing
    timeout=600,
    volumes={VOL_MOUNT: volume},
)
def compare(
    distilled_run: str = "default",
):
    """Phase 4: evaluate teacher, baseline, and distilled student side by side."""
    import os
    import sys
    _setup_dirs()

    teacher_ckpt   = f"{CKPT_DIR}/teacher.pth"
    baseline_ckpt  = f"{CKPT_DIR}/student_baseline.pth"
    distilled_ckpt = f"{CKPT_DIR}/student_distilled_{distilled_run}.pth"

    for name, p in [("teacher", teacher_ckpt),
                    ("baseline", baseline_ckpt),
                    ("distilled", distilled_ckpt)]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{name} checkpoint missing at {p}.\n"
                f"Make sure you've run all previous phases."
            )

    sys.argv = [
        "compare.py",
        "--teacher",   teacher_ckpt,
        "--baseline",  baseline_ckpt,
        "--distilled", distilled_ckpt,
        "--data-root", DATA_DIR,
    ]
    import compare as compare_mod
    compare_mod.main()


# ─────────────────────────────────────────────────────────────────────
# CONVENIENCE: run all four phases sequentially
# ─────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def run_all(
    epochs: int = 200,
    temperature: float = 4.0,
    alpha: float = 0.7,
    beta: float = 100.0,
):
    """Run the full pipeline (all four phases) end-to-end."""
    print(">>> Phase 1: training teacher")
    train_teacher.remote(epochs=epochs)

    print("\n>>> Phase 2: training baseline student")
    train_baseline.remote(epochs=epochs)

    print("\n>>> Phase 3: distillation")
    distill.remote(
        run_name="default",
        epochs=epochs,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
    )

    print("\n>>> Phase 4: comparison")
    compare.remote(distilled_run="default")

    print("\n[✓] Full pipeline complete.")
    print("Download the distilled checkpoint locally with:")
    print("  modal volume get kd-results /checkpoints/student_distilled_default.pth ./")


# ─────────────────────────────────────────────────────────────────────
# ABLATION: run multiple distillation configs in parallel
# ─────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def ablation(epochs: int = 100):
    """
    Run 6 distillation configs in parallel on separate Modal workers.

    Requires teacher.pth and student_baseline.pth already on the volume.
    Use shorter epochs (default 100) since this fans out 6× resources.
    """
    configs = [
        # (run_name,    T,   α,   β)
        ("no_kd",      1.0, 0.0, 0.0),     # control: pure CE training
        ("logit_T2",   2.0, 0.7, 0.0),     # logit KD, low temperature
        ("logit_T4",   4.0, 0.7, 0.0),     # logit KD, default temperature
        ("logit_T8",   8.0, 0.7, 0.0),     # logit KD, high temperature
        ("feat_only",  4.0, 0.0, 100.0),   # feature KD only
        ("full",       4.0, 0.7, 100.0),   # everything combined
    ]

    print(f"Launching {len(configs)} distillation runs in parallel...")

    # Modal's .map() spawns each call on its own worker
    args_list = [
        {
            "run_name":    name,
            "epochs":      epochs,
            "temperature": T,
            "alpha":       a,
            "beta":        b,
        }
        for (name, T, a, b) in configs
    ]

    # starmap_async would also work; we use a sync for-loop with .spawn()
    handles = [distill.spawn(**kw) for kw in args_list]
    results = [h.get() for h in handles]

    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)
    for cfg, res in zip(configs, results):
        name, T, a, b = cfg
        print(f"  {name:12s}  T={T}  α={a}  β={b}  →  {res['checkpoint']}")

    print("\nTo evaluate any of them:")
    print("  modal run modal_app.py::compare --distilled-run <run_name>")


# ─────────────────────────────────────────────────────────────────────
# UTILITY: list checkpoints on the volume
# ─────────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    timeout=60,
    volumes={VOL_MOUNT: volume},
)
def list_checkpoints():
    """List all checkpoints currently on the Modal volume."""
    import os
    _setup_dirs()
    print("Checkpoints on volume:")
    for f in sorted(os.listdir(CKPT_DIR)):
        path = os.path.join(CKPT_DIR, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {f:50s}  {size_mb:6.2f} MB")

    print("\nLogs:")
    if os.path.exists(LOG_DIR):
        for f in sorted(os.listdir(LOG_DIR)):
            print(f"  {f}")
