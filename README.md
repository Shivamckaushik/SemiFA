# SemiFA: Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis

**SemiFA** is a multi-modal agentic framework that autonomously generates structured semiconductor failure analysis (FA) reports from inspection images in under one minute — replacing a manual expert review and reporting process that can consume several hours per case.

> **Paper:** *"SemiFA: An Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis Report Generation"*
> Shivam Chand Kaushik — School of Artificial Intelligence & Data Science, IIT Jodhpur

[![arXiv](https://img.shields.io/badge/arXiv-2604.13236-b31b1b.svg)](https://arxiv.org/abs/2604.13236)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shivamckaushik/SemiFA/blob/main/colab/SemiFA_Reproducibility.ipynb)

---

## Overview

SemiFA decomposes the FA process into a four-agent LangGraph pipeline. Each agent is a pure function operating on a shared `FAState`, enabling modular, auditable, and fault-tolerant workflows.

| Agent | Role |
|---|---|
| **DefectDescriber** | DINOv2-base classification (9 classes) + LLaVA-1.6 defect narration |
| **RootCauseAnalyzer** | SECS/GEM telemetry + Qdrant top-5 retrieval + LLaVA hypothesis generation |
| **SeverityClassifier** | LLaVA severity assessment (CRITICAL / MAJOR / MINOR / NONE) + yield impact |
| **RecipeAdvisor** | LLaVA corrective action and process parameter recommendations |
| **ReportGenerator** | PDF + JSON FA report assembly via ReportLab; Qdrant upsert for self-improvement |

Pipeline diagram: [`paper/figures/pipeline_diagram.pdf`](paper/figures/pipeline_diagram.pdf)

---

## Experiments vs Production Deployment

This repository contains two distinct layers:

| Layer | Location | Purpose |
|---|---|---|
| **Experimental validation** | `colab/SemiFA_Reproducibility.ipynb` | Reproduces every number in the paper — DINOv2 training, LLaVA inference, latency benchmark, batch analysis, FA report PDF. Run on Google Colab Pro (A100). No local GPU or Docker required. |
| **Production architecture** | `src/` + `docker-compose.yml` | Reference implementation of the full deployable system — FastAPI, TimescaleDB, MinIO, MQTT, SECS/GEM, Qdrant, Streamlit UI. Designed for real fab integration; requires NVIDIA GPU + Docker. |

**All paper claims are backed by the Colab notebook.** The `src/` layer demonstrates the system architecture and deployment design; it was not exercised during paper experiments (SECS/GEM is in simulator mode by default, MinIO and TimescaleDB are replaced by in-memory equivalents in the notebook).

---

## Key Results

### Pipeline Latency

| Run | Hardware | Decoding | Total |
|---|---|---|---|
| Paper experiments | A100-SXM4-40 GB, PyTorch 2.1.0 | Greedy (`do_sample=False`) | **48.4 s** |
| Reproducibility run | A100-SXM4-40 GB, PyTorch 2.10.0+cu128 | Sampling (`temp=0.3`) | ~65–67 s |

The ~17 s difference is entirely due to generation parameters: sampling-based decoding with repetition penalty explores a larger token search space than greedy decoding. Hardware and model weights are identical.

**Per-node breakdown (paper experiments, median of 3 runs):**

| Agent Node | Latency |
|---|---|
| DefectDescriber | 22.0 s |
| RootCauseAnalyzer | 11.9 s |
| SeverityClassifier | 5.5 s |
| RecipeAdvisor | 6.5 s |
| ReportGenerator | 2.5 s |
| **Total** | **48.4 s** |

Speedup over manual FA: **150–300×**

### Visual Encoder Comparison (SemiFA-930 val set, 140 images)

| Method | Accuracy | Macro F1 | Trainable Params |
|---|---|---|---|
| CLIP ViT-B/32 (zero-shot) | 20.7% | 0.173 | 0 |
| ResNet-50 (end-to-end fine-tune) | 82.9% | 0.839 | 23.5 M |
| **DINOv2-base + MLP (ours)** | **90.0%** | **0.898** | **214 K** |

DINOv2 achieves a 7.1 pp improvement over ResNet-50 with 110× fewer trainable parameters.

**Reproducibility run:** 90.7% (127/140), Macro F1 = 0.903. The 0.7 pp difference is within normal MLP initialisation variance across runs. A dedicated full-training run reported 92.1% (Macro F1 = 0.917).

### Multi-Modal Fusion Ablation (GPT-4o judge, 1–5 composite, 5 cases)

| Condition | Composite Score |
|---|---|
| Full system (visual + telemetry + retrieval) | 3.60 |
| No retrieval (visual + telemetry) | 3.73 |
| No telemetry (visual + retrieval) | 3.33 |
| Image only (baseline) | 2.74 |

Multi-modal fusion improves over image-only baseline by **+0.86 points**. Equipment telemetry is the more load-bearing modality: removing it drops the composite by 0.27 points.

### Post-Submission Reproducibility Run — 6-Image Batch

One image per defect class was selected (random seed 7) and run through the full pipeline independently.

| Image | True Class | Predicted | Match |
|---|---|---|---|
| 1 | scratch | scratch | ✓ |
| 2 | center_cluster | center_cluster | ✓ |
| 3 | ring_pattern | ring_pattern | ✓ |
| 4 | near_full_wafer | near_full_wafer | ✓ |
| 5 | random_defects | random_defects | ✓ |
| 6 | no_defect | no_defect | ✓ |

**Batch accuracy: 6/6 (100%). Total wall time: 292 s (~48.7 s/image).**

---

## Dataset — SemiFA-930

930 annotated semiconductor defect images across 9 defect classes, each paired with a structured FA narrative for VLM fine-tuning.

| Source | Classes | Images |
|---|---|---|
| Procedural synthesis | scratch, particle_contamination, edge_crack | 318 |
| WM-811K | center_cluster, local_cluster, ring_pattern, random_defects, near_full_wafer, no_defect | 572 |
| MixedWM38 | ring_pattern (supplement) | 40 |
| **Total** | **9** | **930** |

Train / val split: **790 / 140**

**Dataset:** [https://huggingface.co/datasets/ShivamChand/SemiFA-930](https://huggingface.co/datasets/ShivamChand/SemiFA-930)

---

## Reproduce Experiments (Colab)

All paper experiments were run on Google Colab Pro (NVIDIA A100-SXM4-40 GB). No local GPU or Docker required.

**Open the reproducibility notebook:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shivamckaushik/SemiFA/blob/main/colab/SemiFA_Reproducibility.ipynb)

**Required: Google Drive folder `semifa/` containing:**
- `semifa_dataset.tar.gz` — dataset archive (download from HuggingFace: [ShivamChand/SemiFA-930](https://huggingface.co/datasets/ShivamChand/SemiFA-930))

**Notebook structure (run cells in order):**

| Cell | Step |
|---|---|
| 1 | GPU check, TF32 enable, global constants |
| 2 | Python imports |
| 3 | Mount Google Drive + extract dataset + verify JSONL counts |
| 4 | Install dependencies (runtime restarts automatically) |
| 5 | Re-run imports after restart |
| 6 | Train DINOv2 MLP head (50 epochs) → reproduces encoder classification results |
| 7 | Evaluate DINOv2 → classification report |
| 8 | Load LLaVA-1.6 in 4-bit NF4 + define `llava_infer()` helper |
| 9 | Verify LLaVA generation |
| 10 | Latency benchmark (median of 3 runs per node) → reproduces pipeline latency table |
| 11 | Verify benchmark results |
| 12 | Full single-image FA pipeline → generates FA report PDF with embedded image |
| 13 | Display report |
| 14 | 6-image batch pipeline → generates combined findings PDF |
| 15 | Display batch summary |
| 16 | QLoRA fine-tuning (optional, requires large dataset) |

The notebook with full cell outputs is committed at `colab/SemiFA_Reproducibility.ipynb`.

---

## Repository Structure

```
├── src/                        # Production system (reference architecture)
│   ├── agents/                 # LangGraph agent nodes + FAState TypedDict
│   ├── vision/                 # DINOv2 encoder + wafer map analyzer
│   ├── models/                 # LLaVA inference wrapper + ModelRegistry singleton
│   ├── data/                   # Qdrant, MinIO, TimescaleDB, MQTT, SECS/GEM clients
│   ├── reports/                # ReportLab PDF generator
│   └── api/                    # FastAPI endpoints + Pydantic schemas
├── training/                   # QLoRA fine-tuning + evaluation scripts
├── experiments/                # Encoder comparison script + results JSON
├── colab/                      # SemiFA_Reproducibility.ipynb (with full outputs)
├── data/processed/             # train.jsonl + val.jsonl (790 + 140 records)
├── paper/figures/              # Pipeline diagram + wafer map grid + sample FA report
├── tests/                      # pytest suite (CPU-only, no live services)
├── infra/                      # SQL schema (TimescaleDB hypertables) + Mosquitto config
├── requirements.txt
├── docker-compose.yml          # Full stack: Qdrant, TimescaleDB, MQTT, MinIO, API, UI
└── .env.example
```

> **Note:** The paper manuscript (`semifa_arxiv.tex` / `.pdf`) is not included in this repository pending journal publication. The `paper/figures/` directory contains the pipeline architecture diagram and wafer map visualisations referenced in the paper.

---

## Local / Docker Deployment

The `src/` architecture is designed for production fab integration. It is not required to reproduce paper results (use the Colab notebook for that).

> **Note:** The full Docker stack has not been end-to-end tested on a local machine by the author. The `src/` code, `docker-compose.yml`, and service configurations represent the intended production architecture and are provided as a reference implementation. Individual components (FastAPI routes, agent nodes, vision modules) are covered by the `tests/` suite (CPU-only, mocked services). End-to-end local deployment requires an NVIDIA GPU, Docker with NVIDIA Container Toolkit, and all credentials in `.env`.

```bash
# Clone
git clone https://github.com/Shivamckaushik/SemiFA.git
cd SemiFA

# Copy and fill environment variables
cp .env.example .env
# Set HF_TOKEN in .env for LLaVA-1.6 weights download

# Start infrastructure (Qdrant, TimescaleDB, MinIO, Mosquitto)
docker compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Start API (hot-reload)
uvicorn src.api.main:app --reload --port 8000

# Start Streamlit frontend
streamlit run frontend/app.py --server.port 8501
```

Requires: Docker with NVIDIA Container Toolkit, NVIDIA GPU with CUDA, HuggingFace token (`HF_TOKEN`).

---

## Run Tests

```bash
pytest tests/ -v   # CPU-only, no GPU or live services required
```

---

## Generate Synthetic Dataset

```bash
python scripts/generate_synthetic_dataset.py
python scripts/download_wm811k.py
python scripts/download_mixedwm38.py
python scripts/build_semifa_dataset.py
```

---

## Citation

If you use SemiFA or SemiFA-930, please cite:

```bibtex
@article{kaushik2026semifa,
  title     = {{SemiFA}: An Agentic Multi-Modal Framework for Autonomous
               Semiconductor Failure Analysis Report Generation},
  author    = {Kaushik, Shivam Chand},
  journal   = {arXiv preprint arXiv:2604.13236},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.13236}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
