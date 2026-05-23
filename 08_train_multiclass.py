import pandas as pd
from datasets import Dataset, Features, ClassLabel, Value
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
import sys
import json
import shutil
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Paths and config values ---
DATA_DIR = r'./data'
TRAINING_DATA_FILE = os.path.join(DATA_DIR, 'model_only_classification_result_final.xlsx')
OUTPUT_BASE_DIR = r'./multiclass_model'
RESULTS_DIR = os.path.join(OUTPUT_BASE_DIR, 'results')
MODEL_SAVE_DIR = os.path.join(OUTPUT_BASE_DIR, 'trained_model')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# --- 2. Data loading and preprocessing ---
try:
    MIN_SAMPLES_PER_CLASS = 20
    df = pd.read_excel(TRAINING_DATA_FILE, sheet_name='Processed_dataset')

    label_counts = df['Predicted_Label'].value_counts()
    labels_to_keep = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()

    df_train = df[df['Predicted_Label'].isin(labels_to_keep)].reset_index(drop=True)
    df_train = df_train[['service_title_cleaned', 'Predicted_Label']].rename(columns={'service_title_cleaned': 'text', 'Predicted_Label': 'label'})
    df_train.dropna(inplace=True)

    # --- [core fix 1] Global label mapping definition ---
    logger.info("Global label mapping create start...")

    # Sort string labels for consistent ordering
    unique_labels_str = sorted(df_train['label'].unique().tolist())

    # Create label2id and id2label
    label2id_global = {label: i for i, label in enumerate(unique_labels_str)}
    # id2label uses 'int keys' per Hugging Face config standard.
    # On save it is auto-converted to string keys, so no concern.
    id2label_global = {i: label for i, label in enumerate(unique_labels_str)}

    # Convert df_train's 'label' column to int IDs using the mapping
    df_train['label'] = df_train['label'].map(label2id_global)

    num_labels_global = len(unique_labels_str)

    logger.info(f"Global label map created. Total classes: {num_labels_global}")
    logger.info(f"Label to ID Mapping: {label2id_global}")

except Exception as e:
    logger.error(f"[FATAL] Data load or preprocessing failed: {e}", exc_info=True)
    sys.exit(1)


# --- 3. Class imbalance handling: compute weights ---
class_weights = compute_class_weight('balanced', classes=np.unique(df_train['label']), y=df_train['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float)
logger.info(f"Class imbalance correction weights computed. (Shape: {class_weights.shape})")

# --- 4. Model, tokenizer, training-related class and function definitions ---
MODEL_NAME = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenizer config
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# FocalLoss class
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Combined trainer
class UltimateTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights=None,
        rdrop_alpha=0.5,
        adv_alpha=0.5,
        adv_epsilon=1.0,
        focal_loss_gamma=2.0,
        calibration_loader=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha
        self.adv_alpha = adv_alpha
        self.adv_epsilon = adv_epsilon
        # Move class weight tensor to the same device for GPU use
        weights = class_weights.to(self.args.device) if class_weights is not None else None
        # Init FocalLoss with weights
        self.focal_loss_fct = FocalLoss(weight=weights, gamma=focal_loss_gamma)
        self.calibration_loader = calibration_loader

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not model.training:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            return (loss, outputs) if return_outputs else loss

        labels = inputs.get("labels")

        # 1. Original Forward Pass (for R-Drop)
        outputs1 = model(**inputs)
        # 2. Second Forward Pass (for R-Drop)
        outputs2 = model(**inputs)

        # 3. Base loss and R-Drop loss compute
        loss1 = self.focal_loss_fct(outputs1.logits, labels)
        loss2 = self.focal_loss_fct(outputs2.logits, labels)
        loss_ce_avg = (loss1 + loss2) / 2

        kl_loss = F.kl_div(
            F.log_softmax(outputs1.logits, dim=-1),
            F.log_softmax(outputs2.logits, dim=-1),
            reduction='batchmean',
            log_target=True
        )

        # 4. Adversarial Training (FGM): needs the original embeddings.
        embedding_layer = model.get_input_embeddings()
        original_embeddings = embedding_layer(inputs['input_ids'])

        # Compute gradient on original embeddings. Trainer should not call backward twice;
        # use adv_loss only to construct the perturbation, then run actual backward on final loss.
        original_embeddings.requires_grad_()

        # forward pass for gradient compute
        temp_inputs = {k: v for k, v in inputs.items() if k != 'input_ids' and k != 'labels'}
        outputs_for_grad = model(inputs_embeds=original_embeddings, labels=labels, **temp_inputs)
        loss_for_grad = self.focal_loss_fct(outputs_for_grad.logits, labels)

        # This gradient is used only to construct the adversarial perturbation.
        grad = torch.autograd.grad(loss_for_grad, original_embeddings, retain_graph=False)[0]

        # compute perturbation (delta)
        delta = self.adv_epsilon * grad / (grad.norm() + 1e-8)

        # briefly turn off model's requires_grad.
        model.eval()
        with torch.no_grad():
            adv_outputs = model(inputs_embeds=original_embeddings.detach() + delta, **temp_inputs)
            loss_adv = self.focal_loss_fct(adv_outputs.logits, labels)
        model.train()

        # 5. combine final loss
        loss = loss_ce_avg + self.rdrop_alpha * kl_loss + self.adv_alpha * loss_adv

        return (loss, outputs1) if return_outputs else loss


    def train(self, *args, **kwargs):
        # run base training first
        train_result = super().train(*args, **kwargs)
        return train_result


    def predict(self, *args, **kwargs):
        # run base predict
        prediction_output = super().predict(*args, **kwargs)
        return prediction_output

# Assessment metric definition
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}


# Cross-validation and training
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
all_eval_results = []
best_f1_overall = -1.0
path_of_best_model_so_far = ""

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['label'])):
    print(f"===================== FOLD {fold+1}/{n_splits} =====================")

    fold_output_dir = os.path.join(RESULTS_DIR, f'fold_{fold+1}')

    train_df_fold = df_train.iloc[train_idx]
    eval_df_fold = df_train.iloc[val_idx]
    train_dataset = Dataset.from_pandas(train_df_fold)
    eval_dataset = Dataset.from_pandas(eval_df_fold)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels_global,
        id2label=id2label_global,
        label2id=label2id_global
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["query", "value"],
        lora_dropout=0.1, bias="none", task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        gradient_accumulation_steps=2,
        fp16=True,
        logging_strategy="epoch",
        report_to="none",
        save_total_limit=1
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier" not in n and "lora" not in n], 'weight_decay': training_args.weight_decay, 'lr': 2e-6},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "classifier" not in n and "lora" not in n], 'weight_decay': 0.0, 'lr': 2e-6},
        {'params': [p for n, p in model.named_parameters() if "classifier" in n or "lora" in n], 'weight_decay': training_args.weight_decay, 'lr': 1e-4}
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    num_training_steps = (len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * training_args.warmup_ratio),
        num_training_steps=num_training_steps
    )

    # Trainer config
    trainer = UltimateTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),
        class_weights=class_weights,
        rdrop_alpha=0.5,
        adv_alpha=0.5,
        focal_loss_gamma=2.0
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Fold {fold+1} assessment result: {eval_results}")
    all_eval_results.append(eval_results)
    current_fold_f1 = eval_results["eval_f1"]

    if current_fold_f1 > best_f1_overall:
        best_f1_overall = current_fold_f1

        # 1. Delete previous best model folder if it existed (safe delete)
        if path_of_best_model_so_far and os.path.exists(path_of_best_model_so_far):
            logger.info(f"Deleting previous best model folder '{os.path.basename(path_of_best_model_so_far)}'.")
            shutil.rmtree(path_of_best_model_so_far, ignore_errors=True)

        # 2. Configure final save path of the new best model
        new_best_model_path = os.path.join(MODEL_SAVE_DIR, f'best_model_fold_{fold+1}')
        path_of_best_model_so_far = new_best_model_path

        # 3. Save the loaded best model directly via trainer
        logger.info(f"*** New overall best performance found! [Fold {fold+1}] -> saving to '{new_best_model_path}'. ***")
        trainer.save_model(new_best_model_path)
        tokenizer.save_pretrained(new_best_model_path)

    # Memory cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


print("\n===================== Final Cross-Validation Results =====================")
avg_accuracy = np.mean([res['eval_accuracy'] for res in all_eval_results])
avg_f1 = np.mean([res['eval_f1'] for res in all_eval_results])
std_f1 = np.std([res['eval_f1'] for res in all_eval_results])

print(f"Mean accuracy: {avg_accuracy:.4f}")
print(f"Mean Macro F1-Score: {avg_f1:.4f} (std: {std_f1:.4f})")
print(f"Best model saved to '{path_of_best_model_so_far}'.")
