# SemiFA: Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis

**SemiFA** is a multi-modal agentic framework that autonomously generates structured semiconductor failure analysis (FA) reports from inspection images in under one minute — replacing a 2–4 hour manual engineering process.

> **Paper:** *"SemiFA: An Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis Report Generation"*
> Shivam Chand Kaushik — School of Artificial Intelligence & Data Science, IIT Jodhpur

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shivamckaushik/SemiFA/blob/main/colab/Final_op_qlora_finetune_colab.ipynb)

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

## Key Results

### Pipeline Latency (NVIDIA A100-SXM4-40 GB, PyTorch 2.1.0, LLaVA-1.6 4-bit NF4)

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

DINOv2 achieves a 7.1 pp improvement over ResNet-50 with 110× fewer trainable parameters. A dedicated full-training run of the DINOv2 head reports 92.1% (129/140, Macro F1 = 0.917).

### Multi-Modal Fusion Ablation (GPT-4o judge, 1–5 composite, 5 cases)

| Condition | Composite Score |
|---|---|
| Full system (visual + telemetry + retrieval) | 3.60 |
| No retrieval (visual + telemetry) | 3.73 |
| No telemetry (visual + retrieval) | 3.33 |
| Image only (baseline) | 2.74 |

Multi-modal fusion improves over image-only baseline by **+0.86 points**. Equipment telemetry (SECS/GEM) is the more load-bearing modality: removing it drops the composite by 0.27 points.

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

## Reproduce Experiments

All experiments were run on Google Colab (NVIDIA A100-SXM4-40 GB, Colab Pro+). No local GPU or Docker required.

**Open the notebook:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shivamckaushik/SemiFA/blob/main/colab/Final_op_qlora_finetune_colab.ipynb)

Run cells in order:
- **Step 1:** Mount Google Drive
- **Step 4:** Install dependencies (runtime restarts automatically)
- **Step 5:** Download SemiFA-930 from HuggingFace
- **Step 9:** Load LLaVA-1.6 in 4-bit NF4
- **Step 10:** Train DINOv2 MLP head → reproduces encoder classification results
- **Step 11:** Measure LLaVA inference latency → reproduces pipeline latency table

The notebook with full cell outputs is committed at `colab/Final_op_qlora_finetune_colab.ipynb`.

### Encoder Comparison (standalone script)

```bash
python experiments/run_encoder_comparison.py \
  --data-root data/processed \
  --epochs 50
```

Results are saved to `experiments/encoder_comparison_results.json`.

---

## Repository Structure

```
├── src/                        # Core system
│   ├── agents/                 # LangGraph agent nodes + FAState TypedDict
│   ├── vision/                 # DINOv2 encoder + wafer map analyzer
│   ├── models/                 # LLaVA inference wrapper + ModelRegistry singleton
│   ├── data/                   # Qdrant, MinIO, TimescaleDB, MQTT, SECS/GEM clients
│   ├── reports/                # ReportLab PDF generator
│   └── api/                    # FastAPI endpoints + Pydantic schemas
├── training/                   # QLoRA fine-tuning + evaluation scripts
├── experiments/                # Encoder comparison script + results JSON
├── scripts/                    # Dataset generation and download scripts
├── colab/                      # Colab notebook with full outputs
├── data/processed/             # train.jsonl + val.jsonl (790 + 140 records)
├── paper/                      # semifa_arxiv.pdf + semifa_arxiv.tex + figures/
├── tests/                      # pytest suite (CPU-only, no live services)
├── infra/                      # SQL schema + Mosquitto config
├── requirements.txt
├── docker-compose.yml          # Full stack (Qdrant, TimescaleDB, MQTT, MinIO, API)
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
# Set HF_TOKEN in .env for LLaVA weights download

# Start infrastructure
docker compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Start API
uvicorn src.api.main:app --reload --port 8000
```

Requires: Docker, NVIDIA GPU with CUDA, HuggingFace token (`HF_TOKEN`) for LLaVA-1.6 weights.

---

## Generate Synthetic Dataset

```bash
python scripts/generate_synthetic_dataset.py
python scripts/download_wm811k.py
python scripts/download_mixedwm38.py
python scripts/build_semifa_dataset.py
```

---

## Run Tests

```bash
pytest tests/ -v   # CPU-only, no GPU or live services required
```

---

## Citation

If you use SemiFA or SemiFA-930, please cite:

```bibtex
@article{kaushik2025semifa,
  title     = {{SemiFA}: An Agentic Multi-Modal Framework for Autonomous
               Semiconductor Failure Analysis Report Generation},
  author    = {Kaushik, Shivam Chand},
  journal   = {arXiv preprint},
  year      = {2025},
  note      = {School of Artificial Intelligence \& Data Science, IIT Jodhpur}
}
```

*arXiv ID will be added upon publication.*

---

## License

MIT License — see [LICENSE](LICENSE) for details.
