# run_final_quantization.py

import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import logging
import sys
import shutil
import subprocess
import pandas as pd
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig
from datasets import Dataset
import onnx

# ==============================================================================
# 1. Init config
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

BASE_MODEL_NAME = "klue/roberta-large"
BINARY_MODEL_DIR = './trained_model/best_model_fold_1'
MULTICLASS_MODEL_DIR = './multiclass_model/trained_model/best_model'

TEMP_ARTIFACTS_DIR = './temp_quantization_artifacts'
FINAL_QUANTIZED_DIR = './final_quantized_onnx_models'

CALIBRATION_DATA_PATH = r'./data/calibration.parquet'
CALIBRATION_TEXT_COLUMN = 'notice_name'
NUM_CALIBRATION_SAMPLES = 200

# ==============================================================================
# 2. Validated model loading functions
# ==============================================================================
def load_binary_classifier():
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)
    peft_model = PeftModel.from_pretrained(base_model, BINARY_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BINARY_MODEL_DIR)
    return peft_model, tokenizer

def load_multiclass_classifier():
    config = AutoConfig.from_pretrained(MULTICLASS_MODEL_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, config=config)
    peft_model = PeftModel.from_pretrained(base_model, MULTICLASS_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MULTICLASS_MODEL_DIR)
    return peft_model, tokenizer

# ==============================================================================
# 3. ONNX model input verify function
# ==============================================================================
def get_onnx_model_inputs(onnx_model_path):
    """Verify which inputs the ONNX model actually requires."""
    model = onnx.load(os.path.join(onnx_model_path, "model.onnx"))
    input_names = [input.name for input in model.graph.input]
    logging.info(f"ONNX model inputs: {input_names}")
    return input_names

# ==============================================================================
# 4. Final orchestration pipeline
# ==============================================================================
def run_pipeline_for_model(model_type_name: str, load_logic: callable):
    logging.info(f"--- pipeline start: {model_type_name} ---")

    # --- Step 1 & 2 ---
    temp_merged_pytorch_path = os.path.join(TEMP_ARTIFACTS_DIR, f"{model_type_name}_pytorch_fp32")
    logging.info(f"1. Load model, merge, and save temp -> path: {temp_merged_pytorch_path}")
    try:
        peft_model, tokenizer = load_logic()
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(temp_merged_pytorch_path)
        tokenizer.save_pretrained(temp_merged_pytorch_path)
        del peft_model, merged_model
    except Exception as e:
        logging.error(f"Step 1 (merge/save) failed: {e}", exc_info=True)
        return False

    temp_fp32_onnx_path = os.path.join(TEMP_ARTIFACTS_DIR, f"{model_type_name}_onnx_fp32")
    command = ["optimum-cli", "export", "onnx", "--model", temp_merged_pytorch_path, "--task", "text-classification", temp_fp32_onnx_path]
    logging.info(f"2. Export to FP32 ONNX... command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        logging.error(f"Step 2 (ONNX export) failed! STDERR:\n{e.stderr}")
        return False

    # --- Step 3: INT8 static quantization (final fix) ---
    final_int8_onnx_path = os.path.join(FINAL_QUANTIZED_DIR, model_type_name)
    logging.info(f"3. INT8 static quantization -> final path: {final_int8_onnx_path}")
    try:
        quantizer = ORTQuantizer.from_pretrained(temp_fp32_onnx_path)
        tokenizer_for_calib = AutoTokenizer.from_pretrained(temp_merged_pytorch_path)

        # Verify which inputs the ONNX model actually requires
        required_inputs = get_onnx_model_inputs(temp_fp32_onnx_path)

        logging.info("  - (3a) Building calibration dataset...")
        calib_df = pd.read_parquet(CALIBRATION_DATA_PATH)
        calib_texts_list = calib_df[CALIBRATION_TEXT_COLUMN].dropna().sample(NUM_CALIBRATION_SAMPLES, random_state=42).tolist()

        # Preprocess data to match the ONNX model's input
        def preprocess_function(examples):
            tokenized = tokenizer_for_calib(examples["text"], padding="max_length", truncation=True, max_length=128)

            # Include only inputs that the ONNX model requires
            result = {}
            if "input_ids" in required_inputs:
                result["input_ids"] = tokenized["input_ids"]
            if "attention_mask" in required_inputs:
                result["attention_mask"] = tokenized["attention_mask"]
            if "token_type_ids" in required_inputs:
                # RoBERTa typically does not use token_type_ids; add only if required
                result["token_type_ids"] = [[0] * len(ids) for ids in tokenized["input_ids"]]

            return result

        calibration_dataset_raw = Dataset.from_dict({"text": calib_texts_list})
        calibration_dataset_processed = calibration_dataset_raw.map(preprocess_function, batched=True, remove_columns=["text"])

        # Use numpy format to ensure compatibility with ONNX Runtime
        calibration_dataset_final = calibration_dataset_processed.with_format("numpy")

        logging.info(f"  - Calibration dataset columns: {calibration_dataset_final.column_names}")

        calibration_config = AutoCalibrationConfig.minmax(calibration_dataset_final)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False)

        logging.info("  - (3b) Running fit() to compute the activation range...")
        calibration_tensors_range = quantizer.fit(
            dataset=calibration_dataset_final,
            calibration_config=calibration_config,
            operators_to_quantize=qconfig.operators_to_quantize
        )

        logging.info("  - (3c) Running quantize() to quantize the model...")
        quantizer.quantize(
            save_dir=final_int8_onnx_path,
            calibration_tensors_range=calibration_tensors_range,
            quantization_config=qconfig
        )

        tokenizer_for_calib.save_pretrained(final_int8_onnx_path)

    except Exception as e:
        logging.error(f"Step 3 (INT8 quantization) failed: {e}", exc_info=True)
        return False

    logging.info(f"--- pipeline success: {model_type_name} ---")
    return True

# ==============================================================================
# 5. Main run block
# ==============================================================================
if __name__ == "__main__":
    try:
        import pandas, optimum, datasets, onnx
    except ImportError:
        print("Error: required libraries are not installed. Run `pip install optimum[onnxruntime] peft pandas torch transformers datasets onnx`.")
        sys.exit(1)

    os.makedirs(FINAL_QUANTIZED_DIR, exist_ok=True)
    if os.path.exists(TEMP_ARTIFACTS_DIR):
        try:
            shutil.rmtree(TEMP_ARTIFACTS_DIR)
        except Exception:
            pass
    os.makedirs(TEMP_ARTIFACTS_DIR, exist_ok=True)

    success_binary = run_pipeline_for_model('binary_classifier', load_binary_classifier)
    success_multiclass = run_pipeline_for_model('multiclass_classifier', load_multiclass_classifier)

    try:
        shutil.rmtree(TEMP_ARTIFACTS_DIR)
        logging.info(f"Temp artifacts folder removed: {TEMP_ARTIFACTS_DIR}")
    except Exception as e:
        logging.warning(f"Temp folder remove failed: {e}")

    if success_binary and success_multiclass:
        print(f"\nAll model ONNX quantization completed successfully.")
        print(f"Final quantized ONNX models saved to: '{FINAL_QUANTIZED_DIR}'")
    else:
        print(f"\nSome or all model quantization failed. Verify logs.")
