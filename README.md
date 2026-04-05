# SemiFA: Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis

**SemiFA** is a multi-modal agentic framework that autonomously generates structured semiconductor failure analysis (FA) reports from inspection images in under one minute — replacing a 2–4 hour manual engineering process.

> Paper: *"SemiFA: An Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis Report Generation"*
> Shivam Chand Kaushik — School of Artificial Intelligence & Data Science, IIT Jodhpur

---

## Overview

SemiFA decomposes the FA process into a four-agent LangGraph pipeline:

| Agent | Role |
|---|---|
| **DefectDescriber** | DINOv2-base classification + LLaVA-1.6 defect narration |
| **RootCauseAnalyzer** | SECS/GEM telemetry + Qdrant retrieval + LLaVA hypothesis generation |
| **SeverityClassifier** | LLaVA severity assessment (CRITICAL / MAJOR / MINOR / NONE) |
| **RecipeAdvisor** | LLaVA corrective action and process parameter recommendations |
| **ReportGenerator** | PDF + JSON report assembly via ReportLab |

![Pipeline Architecture](paper/figures/pipeline_diagram.png)

---

## Key Results

| Metric | Value |
|---|---|
| Defect classification accuracy | **92.1%** (129/140 validation images) |
| Macro F1 | **0.917** |
| Full pipeline latency | **48 seconds** (NVIDIA A100-SXM4-40 GB) |
| Speedup over manual FA | **150–300×** |

---

## Dataset — SemiFA-930

930 annotated semiconductor defect images across 9 defect classes, each paired with a structured FA narrative for VLM fine-tuning.

| Source | Classes | Images |
|---|---|---|
| Procedural synthesis | scratch, particle_contamination, edge_crack | 318 |
| WM-811K | center_cluster, local_cluster, ring_pattern, random_defects, near_full_wafer, no_defect | 572 |
| MixedWM38 | ring_pattern (supplement) | 40 |
| **Total** | **9** | **930** |

**Dataset available at:** [https://huggingface.co/datasets/ShivamChand/SemiFA-930](https://huggingface.co/datasets/ShivamChand/SemiFA-930)

---

## Reproduce Experiments

All experiments were run on Google Colab (NVIDIA A100-SXM4-40 GB, free tier).
No local GPU or Docker required.

**Step 1 — Open the notebook:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shivamckaushik/SemiFA/blob/main/colab/op_qlora_finetune_colab.ipynb)

**Step 2 — Run in order:**
- Step 1: Mount Google Drive
- Step 4: Install dependencies (runtime restarts automatically)
- Step 5: Download dataset
- Step 9: Load LLaVA-1.6 (4-bit NF4)
- Step 10: Train DINOv2 MLP head → produces Table IV results
- Step 11: Measure LLaVA inference latency → produces Table VI results

The notebook with real outputs is available at `colab/Final_op_qlora_finetune_colab.ipynb`.

---

## Repository Structure

```
├── src/                        # Core system (agents, vision, API, reports)
│   ├── agents/                 # LangGraph agent nodes + FAState
│   ├── vision/                 # DINOv2 encoder + wafer map analyzer
│   ├── models/                 # LLaVA inference wrapper + ModelRegistry
│   ├── data/                   # Qdrant, MinIO, TimescaleDB, MQTT clients
│   ├── reports/                # ReportLab PDF generator
│   └── api/                    # FastAPI endpoints
├── training/                   # QLoRA fine-tuning + evaluation scripts
├── scripts/                    # Dataset generation and download scripts
├── colab/                      # Colab notebooks (clean + with outputs)
├── data/processed/             # train.jsonl + val.jsonl (790 + 140 records)
├── paper/                      # semifa_arxiv.pdf + figures
├── requirements.txt
├── docker-compose.yml          # Full stack (Qdrant, TimescaleDB, MQTT, API)
└── .env.example
```

---

## Installation (Local / Docker)

```bash
# Clone
git clone https://github.com/Shivamckaushik/SemiFA.git
cd SemiFA

# Copy and fill environment variables
cp .env.example .env

# Start infrastructure (Qdrant, TimescaleDB, Mosquitto, MinIO)
docker compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Start API
uvicorn src.api.main:app --reload --port 8000
```

Requires: Docker, NVIDIA GPU with CUDA, HuggingFace token (`HF_TOKEN` in `.env`) for LLaVA weights.

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

If you use SemiFA or SemiFA-930 in your work, please cite:

```bibtex
@article{kaushik2024semifa,
  title     = {{SemiFA}: An Agentic Multi-Modal Framework for Autonomous
               Semiconductor Failure Analysis Report Generation},
  author    = {Kaushik, Shivam Chand},
  journal   = {arXiv preprint},
  year      = {2024},
  note      = {School of Artificial Intelligence \& Data Science, IIT Jodhpur}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
