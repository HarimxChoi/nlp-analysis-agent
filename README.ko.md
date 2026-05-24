# nlp-analysis-agent

[English](./README.md) | 한국어

한국 공공조달 공고 NLP. 약지도 학습 기반 입찰가능성 + 카테고리 분류.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What it does

- Binary classifier: 이 공고에 입찰할지 말지 판정 (RoBERTa-large + LoRA + UltimateTrainer)
- Multiclass classifier: 어느 엔지니어링 카테고리인지 분류 (SBERT 약라벨 + LoRA ensemble)
- ONNX INT8 배포: 약 330 MB, CPU 추론 약 50 ms

## Why

공공조달 공고 텍스트는 정형화되어 있지만 도메인 특화임. 일반 BERT는 엔지니어링 ontology를 못 잡음. 수만 건 공고를 사람이 직접 라벨링하는 건 불가능.

## Approach

Cross-task knowledge distillation: binary "입찰가능성" 모델이 먼저 도메인을 학습한 뒤, 그 임베딩으로 클러스터링 + 약라벨링을 돌려 multiclass student를 만든다.

```
Raw notices
  -> 01_preprocess.py            # 1:1 balanced dataset 구축
  -> 02_train_binary.py          # RoBERTa+LoRA, UltimateTrainer (Focal+R-Drop+FGM)
  -> 03_vector_db.py             # SBERT + FAISS index (optional)
  -> 04_cluster.py               # [CLS] -> UMAP + HDBSCAN
  -> 05_ontology.py              # 클러스터별 TF-IDF top keywords
  -> 06_weak_label.py            # Hybrid SBERT (0.9) + LoRA (0.1) ensemble
  -> 06_weak_label_llm.py        # Gemini LLM judge (optional)
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

- `UltimateTrainer`: LoRA fine-tuning 위에 FocalLoss + R-Drop (dual forward KL) + FGM adversarial perturbation을 얹음.
- 약라벨링은 SBERT (general Korean) + fine-tuned RoBERTa (domain)를 per-label max-sim으로 ensemble. 최빈 카테고리에는 hard-rule override layer 추가.
- Static INT8 quantization은 `optimum-cli export onnx` 후 `ORTQuantizer.fit(...)`으로 calibration sample 200개 돌림. per-tensor (non per-channel), AVX512-VNNI config.

## Repository note

코드와 파이프라인 구조만 공개함. 학습된 모델 weight, 학습 데이터는 의도적으로 제외. 특정 조달 데이터셋과 도메인 라벨링에서 파생된 자산임.

## License

MIT
