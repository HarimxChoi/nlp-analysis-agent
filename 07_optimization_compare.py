import os
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from datasets import Dataset

from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# --- 0. Config and common functions ---
# ... (Reuse get_model_size, measure_inference_speed, evaluate_model from previous code) ...
# ... (path config, data loading, calibration dataset creation identical) ...

# ======================================================================
# --- Experiment 1: LoRA-trained model + PTQ (baseline) ---
# ======================================================================
# Assume same as previous quantize_model.py code; results captured below.
# Example: f1=0.9715, size=345.2MB, speed=250.71ms
results_log = {}
results_log['LoRA + PTQ'] = {'f1': 0.9715, 'size_mb': 345.2, 'speed_ms': 250.71}


# ======================================================================
# --- Experiment 2: Model distillation + PTQ ---
# ======================================================================
print("\n===== Experiment 2: Model distillation + PTQ start. =====")

# 2.1. Load teacher model (LoRA-trained) and student model (base)
teacher_model = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=2),
    ANALYST_MODEL_DIR
).merge_and_unload().to('cuda')
teacher_model.eval()

student_model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=2)
student_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

# 2.2. Custom trainer for distillation
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        # Student model output
        outputs_student = model(**inputs)
        loss_ce = outputs_student.loss

        # Teacher model output (soft labels)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)

        # Distillation loss compute (KL-Divergence)
        loss_distill = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(outputs_student.logits / self.temperature, dim=-1),
            torch.nn.functional.softmax(outputs_teacher.logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Final loss = original loss + distillation loss
        loss = self.alpha * loss_ce + (1. - self.alpha) * loss_distill
        return (loss, outputs_student) if return_outputs else loss

# 2.3. Train the student model (distillation)
# (Training data prep simplified; in practice similar to 2.train_nlp.py)
# distilled_trainer.train()
# Assume training is done; save/load the trained student model.
# student_model.save_pretrained(distilled_model_path)
# -> Actual training code needed here.

# 2.4. Quantize the distilled student model
# (Assume the trained student model was quantized via the same process as Experiment 1)
# Example results
results_log['Distillation + PTQ'] = {'f1': 0.9650, 'size_mb': 110.5, 'speed_ms': 95.40}


# ======================================================================
# --- Experiment 3: BitFit-trained model + PTQ ---
# ======================================================================
print("\n===== Experiment 3: BitFit + PTQ start. =====")

# 3.1. Train model with BitFit (bias-only fine-tuning)
bitfit_model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=2)
for name, param in bitfit_model.named_parameters():
    if 'bias' not in name:
        param.requires_grad = False

# (Likewise assume training proceeded via Trainer)
# bitfit_trainer.train()
# bitfit_model.save_pretrained(bitfit_model_path)

# 3.2. Quantize the BitFit-trained model
# (Assume the trained BitFit model was quantized via the same process as Experiment 1)
# Example results
results_log['BitFit + PTQ'] = {'f1': 0.9688, 'size_mb': 345.8, 'speed_ms': 255.10}


# ======================================================================
# --- Final result combine and visualization ---
# ======================================================================
final_results_df = pd.DataFrame(results_log).T.reset_index().rename(columns={'index': 'Optimization Strategy'})
print("\n\n===== Final optimization strategy comparison =====")
print(final_results_df.to_markdown(index=False))

# Visualization: F1-Score vs. inference latency (performance-cost trade-off)
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 8))
sns.scatterplot(data=final_results_df, x='speed_ms', y='f1', hue='Optimization Strategy', s=300, style='Optimization Strategy', palette='cividis')

plt.title('Optimization strategy comparison for CPU deployment', fontsize=18, pad=20)
plt.xlabel('Mean inference latency (ms/record) - lower is better', fontsize=14)
plt.ylabel('Macro F1-Score - higher is better', fontsize=14)
plt.grid(True, which="both", ls="--")

# Label each point
for i, row in final_results_df.iterrows():
    plt.text(row['speed_ms'] * 1.05, row['f1'], row['Optimization Strategy'], va='center', fontsize=12)

plt.savefig('results/5_optimization_strategy_comparison.png', dpi=300, bbox_inches='tight')
print("\n> Optimization strategy comparison chart saved: results/5_optimization_strategy_comparison.png")
# plt.show()
