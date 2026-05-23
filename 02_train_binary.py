import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# data loading and preprocessing
data_path = r'./data/preprocessed_balanced_dataset.csv'
output_base_dir = r'.'
results_dir = os.path.join(output_base_dir, 'results')
model_save_dir = os.path.join(output_base_dir, 'trained_model')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

encodings = ['cp949', 'euc-kr', 'utf-8']
df = None
for encoding in encodings:
    try:
        df = pd.read_csv(data_path, encoding=encoding)
        print(f"Loaded file with {encoding} encoding.")
        break
    except UnicodeDecodeError:
        print(f"{encoding} encoding load failed. Trying next...")
if df is None:
    raise ValueError(f"All encodings failed: {', '.join(encodings)}")

print("Data sample:")
print(df.head())
print("Columns:", df.columns)

# verify class distribution and compute weights
class_distribution = df['label'].value_counts(normalize=True)
print("Class distribution:", class_distribution)
if abs(class_distribution.get(0, 0) - class_distribution.get(1, 0)) > 0.1:
    class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights applied:", class_weights)
else:
    class_weights = None

# tokenizer and tokenization function
model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['service_title'], padding="max_length", truncation=True, max_length=128)

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


# R-Drop Trainer
class RDropTrainer(Trainer):
    def __init__(self, *args, kl_loss_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_loss_weight = kl_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_ce, outputs1 = super().compute_loss(model, inputs, return_outputs=True)

        if self.model.training:
            outputs2 = model(**inputs)
            kl_loss = F.kl_div(
                F.log_softmax(outputs1.logits, dim=-1),
                F.log_softmax(outputs2.logits, dim=-1),
                reduction='batchmean',
                log_target=True
            )

            loss = loss_ce + self.kl_loss_weight * kl_loss
        else:
            loss = loss_ce

        return (loss, outputs1) if return_outputs else loss

# Class-weighted trainer (smoothing loss applied)
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, label_smoothing_factor=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        # Move class weight tensor to the same device for GPU use
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.label_smoothing_factor = label_smoothing_factor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Pass weight parameter directly to CrossEntropyLoss
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing_factor
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# Adversarial Trainer
class AdversarialTrainer(Trainer):
    def __init__(self, *args, adv_alpha=0.5, adv_epsilon=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_alpha = adv_alpha
        self.adv_epsilon = adv_epsilon

    def get_input_embeddings_layer(self, model):
        base_model = getattr(model, 'base_model', model)
        if hasattr(base_model, 'get_input_embeddings'):
            return base_model.get_input_embeddings()
        return None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        loss_ce, outputs = super().compute_loss(model, inputs, return_outputs=True)

        embedding_layer = self.get_input_embeddings_layer(model)
        if embedding_layer is None:
            return loss_ce

        # Obtain embedding layer gradient via backprop. Trainer's accelerator handles backward,
        # so compute the gradient using loss_ce. FGM core: input gradient, requires_grad=True.
        embed_inputs = model.get_input_embeddings()(inputs['input_ids'])
        embed_inputs.requires_grad = True

        # run model with new input
        adv_inputs = inputs.copy()
        adv_inputs.pop("input_ids", None)
        outputs_for_grad = model(**adv_inputs, inputs_embeds=embed_inputs)
        loss_for_grad = outputs_for_grad.loss

        # compute gradient
        grad_embeds, = torch.autograd.grad(outputs=loss_for_grad, inputs=embed_inputs)

        # compute perturbation
        delta = self.adv_epsilon * grad_embeds.detach() / (grad_embeds.detach().norm(p=2, dim=-1, keepdim=True) + 1e-8)

        # 3. compute adversarial loss
        with torch.no_grad():
            adv_outputs = model(**adv_inputs, inputs_embeds=(embed_inputs + delta))
            adv_loss = F.cross_entropy(adv_outputs.logits, inputs['labels'])

        # final loss
        loss = loss_ce + self.adv_alpha * adv_loss

        return (loss, outputs) if return_outputs else loss

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
        self.platt_scaler = None
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

        # train Platt Scaling model after training
        self._calibrate_platt_scaler()

        return train_result

    def _calibrate_platt_scaler(self):
        if self.calibration_loader is None:
            print("Warning: No calibration_loader provided. Skipping Platt Scaling.")
            return

        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for inputs in self.calibration_loader:
                inputs = self._prepare_inputs(inputs)
                outputs = self.model(**inputs)
                all_logits.append(outputs.logits.detach().cpu().numpy())
                all_labels.append(inputs['labels'].detach().cpu().numpy())

        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)

        # Assume binary classification; use positive class (1) logits only
        positive_class_logits = all_logits[:, 1].reshape(-1, 1)

        print("Calibrating Platt Scaler...")
        self.platt_scaler = LogisticRegression()
        self.platt_scaler.fit(positive_class_logits, all_labels)
        print("Platt Scaler calibrated.")

    def predict(self, *args, **kwargs):
        # run base predict
        prediction_output = super().predict(*args, **kwargs)

        if self.platt_scaler:
            print("Applying Platt Scaling to predictions...")
            # extract positive class logits (adjust index per model output)
            positive_class_logits = prediction_output.predictions[:, 1].reshape(-1, 1)
            # compute calibrated probabilities
            calibrated_probs = self.platt_scaler.predict_proba(positive_class_logits)

            # Could attach calibrated probabilities to prediction result if needed
            # example: prediction_output.calibrated_predictions = calibrated_probs

        return prediction_output


# assessment metric definition
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}


# Cross-validation and training
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
all_eval_results = []
best_f1 = 0
best_model_path = ""

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"===================== FOLD {fold+1}/{n_splits} =====================")

    train_df = df.iloc[train_idx]
    eval_df = df.iloc[val_idx]
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['service_title'])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=['service_title'])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["query", "value"],
        lora_dropout=0.1, bias="none", task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(results_dir, f'fold_{fold+1}'),
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

    if eval_results['eval_f1'] > best_f1:
        best_f1 = eval_results['eval_f1']
        best_model_path = os.path.join(model_save_dir, f'best_model_{fold+1}')
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"*** new best model saved: {best_model_path} (F1: {best_f1}) ***")

print("\n===================== Final Cross-Validation Results =====================")
avg_accuracy = np.mean([res['eval_accuracy'] for res in all_eval_results])
avg_f1 = np.mean([res['eval_f1'] for res in all_eval_results])
std_f1 = np.std([res['eval_f1'] for res in all_eval_results])

print(f"Mean accuracy: {avg_accuracy:.4f}")
print(f"Mean Macro F1-Score: {avg_f1:.4f} (std: {std_f1:.4f})")
print(f"Best model saved to '{best_model_path}'.")
