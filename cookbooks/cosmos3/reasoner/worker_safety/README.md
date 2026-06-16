# Worker Safety in a Classical Warehouse with Cosmos 3 Reasoner

Classify industrial safety behaviors from warehouse surveillance videos using the
Cosmos 3 Reasoner (VLM). The model performs 8-class safety classification via
structured chain-of-thought reasoning and outputs machine-readable JSON — no
fine-tuning required.

## What You'll Build

- Zero-shot video safety classifier over 8 hazard/safe classes
- Structured JSON output with chain-of-thought reasoning (`<think>` tags)
- Evaluation pipeline across 40 videos with per-class accuracy analysis
- Domain-specific prompt engineering for industrial safety inspection

## Prerequisites

- Backend setup: [vLLM setup](../../README.md#vllm)
- 1× GPU with ≥ 24 GB VRAM for Cosmos3-Nano (1× RTX 4090 / RTX 5000 / A5000)
- 4× H100 80 GB for Cosmos3-Super
- Hugging Face account with accepted [Cosmos3 license](https://huggingface.co/nvidia/Cosmos3-Nano)

### Additional Dependencies

```bash
uv pip install fiftyone openai datasets huggingface_hub
```

## Backends

| Backend | Notebook | GPU Requirement |
|---------|----------|----------------|
| vLLM    | [`run_worker_safety_with_vllm.ipynb`](run_worker_safety_with_vllm.ipynb) | 1× GPU ≥ 24 GB (Nano) / 4× H100 (Super) |

## Quick Start

```bash
# 1. Set up the vLLM backend (see shared setup guide)
# 2. Launch vLLM server
vllm serve nvidia/Cosmos3-Nano \
  --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
  --tensor-parallel-size 1 \
  --allowed-local-media-path "$(pwd)" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --max-model-len 16384 \
  --port 8001

# 3. Run the notebook
jupyter lab run_worker_safety_with_vllm.ipynb
```

## Classification Table

| ID | Label | Definition | Hazard |
|----|-------|-----------|--------|
| 0 | Safe Walkway Violation | Worker walks OUTSIDE the designated Green Path | Unsafe |
| 1 | Unauthorized Intervention | Worker interacts with machine board WITHOUT a green vest | Unsafe |
| 2 | Opened Panel Cover | Machine panel cover is left OPEN after intervention | Unsafe |
| 3 | Carrying Overload with Forklift | Forklift carries 3 OR MORE blocks | Unsafe |
| 4 | Safe Walkway | Worker walks INSIDE the designated Green Path | Safe |
| 5 | Authorized Intervention | Worker interacts with machine board WITH a green vest | Safe |
| 6 | Closed Panel Cover | Machine panel cover is CLOSED after intervention | Safe |
| 7 | Safe Carrying | Forklift carries 2 OR FEWER blocks | Safe |

## Results / Expected Output

Evaluation on the full 40-video [`pjramg/Safe_Unsafe_Test`](https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test) dataset using **Cosmos3-Nano** on a single RTX PRO 5000 (24 GB):

| Metric | Value |
|--------|-------|
| Exact Class Accuracy | 48.7% (19/39) |
| Category Accuracy | 56.4% (22/39) |
| Hazard Detection | 56.4% (22/39) |
| Avg Inference Time | 38.1s per video |

### Per-Class Accuracy

| Class | Accuracy | Notes |
|-------|----------|-------|
| 3 — Forklift Overload | **100%** (5/5) | Perfect detection |
| 7 — Safe Carrying | **100%** (4/4) | Perfect detection |
| 1 — Unauthorized Intervention | **80%** (4/5) | Strong |
| 4 — Safe Walkway | **60%** (3/5) | Moderate |
| 5 — Authorized Intervention | **40%** (2/5) | Distinguishing vest color is challenging |
| 2 — Opened Panel Cover | **20%** (1/5) | Fine-grained state detection is hard for Nano |
| 0 — Walkway Violation | **0%** (0/5) | Path boundary detection needs Super |
| 6 — Closed Panel Cover | **0%** (0/5) | Panel state detection needs Super |

### Successful Predictions — Sample Videos

Below are representative videos where the model correctly classified the safety behavior with chain-of-thought reasoning. All videos are from the open-access [pjramg/Safe_Unsafe_Test](https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test) dataset.

#### Class 1 — Unauthorized Intervention (Unsafe)

<video src="https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test/resolve/main/data/1_tr28.mp4" width="480" controls></video>

> **Prediction:** Class 1 — Unauthorized Intervention ✅
> **Model reasoning:** "A worker in a dark uniform approaches a large industrial machine and interacts with its control panel without wearing a green safety vest, which indicates unauthorized intervention."

#### Class 3 — Carrying Overload with Forklift (Unsafe)

<video src="https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test/resolve/main/data/3_tr14.mp4" width="480" controls></video>

> **Prediction:** Class 3 — Carrying Overload with Forklift ✅
> **Model reasoning:** "A forklift is transporting multiple stacked orange containers across the factory floor, moving from left to right. The stack contains three or more containers, which constitutes an overloaded condition."

#### Class 4 — Safe Walkway (Safe)

<video src="https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test/resolve/main/data/4_tr39.mp4" width="480" controls></video>

> **Prediction:** Class 4 — Safe Walkway ✅
> **Model reasoning:** "A person is walking through the factory aisle while staying within the designated green safety path. The worker follows the marked walkway correctly."

#### Class 5 — Authorized Intervention (Safe)

<video src="https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test/resolve/main/data/5_tr16.mp4" width="480" controls></video>

> **Prediction:** Class 5 — Authorized Intervention ✅
> **Model reasoning:** "A worker in a green safety vest is operating a large industrial machine, pressing buttons on its control panel. The green vest indicates authorized access."

#### Class 7 — Safe Carrying (Safe)

<video src="https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test/resolve/main/data/7_tr3.mp4" width="480" controls></video>

> **Prediction:** Class 7 — Safe Carrying ✅
> **Model reasoning:** "A forklift is moving a single pallet with two stacked orange pallets across a factory floor. The forklift carries 2 or fewer items, which is within the safe limit."

### Expected JSON Output Structure

Each inference returns a structured JSON object after chain-of-thought reasoning:

```json
{
  "prediction_class_id": 3,
  "prediction_label": "Carrying Overload with Forklift",
  "video_description": "A forklift is transporting three stacked orange containers through a factory aisle.",
  "hazard_detection": {
    "is_hazardous": true,
    "temporal_segment": "0.0 - 9.9 seconds"
  }
}
```

### Key Findings

- **Forklift monitoring is production-ready** — 100% accuracy on both overload and safe-carrying classes, even with Cosmos3-Nano on a consumer GPU.
- **Intervention detection is strong** — 80% for unauthorized, with the model correctly identifying vest-wearing workers.
- **Walkway and panel classes need Cosmos3-Super** — these require fine-grained spatial understanding (path boundaries, panel states) that benefits from the larger model.

## Dataset

| Name | Source | License | Size |
|------|--------|---------|------|
| Safe_Unsafe_Test | [pjramg/Safe_Unsafe_Test](https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test) | Apache 2.0 | ~500 MB (40 videos) |
