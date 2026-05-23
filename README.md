# nlp-analysis-agent

Korean public procurement notice NLP — bidability + category classification with weak supervision.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What it does

- Binary classifier: should we bid on this notice? (RoBERTa-large + LoRA + UltimateTrainer)
- Multiclass classifier: which engineering category? (weak labels from SBERT + LoRA ensemble)
- ONNX INT8 deploy: ~330 MB, ~50 ms CPU inference

## Why

Public procurement notice text is structured but domain-specific. Standard BERT does not capture engineering ontology. Manual labeling for tens of thousands of notices is not feasible.

## Approach

Cross-task knowledge distillation: a binary "bidability" model first learns the domain, then its embeddings drive clustering + weak labeling for a multiclass student.

```
Raw notices
  -> 01_preprocess.py            # build balanced 1:1 dataset
  -> 02_train_binary.py          # RoBERTa+LoRA, UltimateTrainer (Focal+R-Drop+FGM)
  -> 03_vector_db.py             # SBERT + FAISS index (optional)
  -> 04_cluster.py               # [CLS] -> UMAP + HDBSCAN
  -> 05_ontology.py              # TF-IDF top keywords per cluster
  -> 06_weak_label.py            # Hybrid SBERT (0.9) + LoRA (0.1) ensemble
  -> 06_weak_label_llm.py        # Optional Gemini-based LLM judge
  -> 07_optimization_compare.py  # LoRA+PTQ vs Distillation+PTQ vs BitFit+PTQ
  -> 08_train_multiclass.py      # K-Fold + class-balanced UltimateTrainer
  -> 09_quantize.py              # LoRA merge -> ONNX FP32 -> INT8 (static calib.)
  -> 10_serve.py                 # FastAPI /classify_batch
```

## Quick Start

```bash
pip install -r requirements.txt

# Binary classifier
python 02_train_binary.py

# Semantic clustering + ontology
python 04_cluster.py
python 05_ontology.py

# Weak labeling
python 06_weak_label.py

# Multiclass classifier
python 08_train_multiclass.py

# Quantize and serve
python 09_quantize.py
python 10_serve.py
```

## Key implementation notes

- `UltimateTrainer` combines FocalLoss + R-Drop (KL on dual forward) + FGM adversarial perturbation, on top of LoRA fine-tuning.
- Weak labeling ensembles SBERT (general Korean) and a fine-tuned RoBERTa (domain) via per-label max-sim, with a hard-rule override layer for the most common categories.
- Static INT8 quantization runs `optimum-cli export onnx` then `ORTQuantizer.fit(...)` over 200 calibration samples, with per-tensor (non per-channel) AVX512-VNNI config.

## Repository note

This repo provides the code and pipeline structure. Trained model weights and training data are intentionally not included; they are derived from a specific procurement dataset with domain-specific labeling.

## License

MIT
