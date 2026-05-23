import os
import logging
import asyncio
from typing import List, Dict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# ==============================================================================
# 1. Logging and base config
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

# --- model path config ---
BASE_MODEL_PATH = r'./quantized_onnx_models'
BINARY_MODEL_DIR = os.path.join(BASE_MODEL_PATH, 'binary_classifier')
MULTI_CLASS_MODEL_DIR = os.path.join(BASE_MODEL_PATH, 'multiclass_classifier')

# ==============================================================================
# 2. FastAPI app and Pydantic definitions
# ==============================================================================
app = FastAPI(
    title="Bid notice analysis AI",
    description="Analyze bid notice text to predict eligibility and service category.",
    version="1.0.0"
)

# --- Input (Request) ---
class NoticeInput(BaseModel):
    id: str = Field(..., description="notice_id", example="20240500001")
    text: str = Field(..., description="text of the notice_name to analyze", example="Strategic environmental impact assessment and climate impact assessment service for a waste-processing facility")

class ClassificationRequest(BaseModel):
    notices: List[NoticeInput]

# --- Output (Response) ---
class ClassificationResult(BaseModel):
    original_id: str
    possibility: float = Field(..., description="final bid eligibility (%)", example=97.4)
    category: str = Field(..., description="final bid category", example="structural_safety_inspection")

# ==============================================================================
# 3. Model loading
# ==============================================================================
models = {}

@app.on_event("startup")
def load_models():
    """Load quantized ONNX models into memory at server startup."""
    logger.info("AI model loading start...")
    try:
        models["binary_tokenizer"] = AutoTokenizer.from_pretrained(BINARY_MODEL_DIR)
        models["binary_model"] = ORTModelForSequenceClassification.from_pretrained(BINARY_MODEL_DIR)

        models["multi_tokenizer"] = AutoTokenizer.from_pretrained(MULTI_CLASS_MODEL_DIR)
        models["multi_model"] = ORTModelForSequenceClassification.from_pretrained(MULTI_CLASS_MODEL_DIR)

        logger.info("AI model loading success.")
    except Exception as e:
        logger.critical(f"Model load failed: {e}", exc_info=True)
# ==============================================================================
# 4. Core business-logic functions
# ==============================================================================

# --- 4a. Single-model inference ---
def run_single_prediction(model, tokenizer, text: str) -> Dict[str, float]:
    """Run inference for a single text with an ONNX model and return the probability distribution."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # Many ONNX models do not accept token_type_ids; remove if present
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']

        with torch.no_grad():
            outputs = model(**inputs)

        if isinstance(outputs.logits, np.ndarray):
            logits = torch.tensor(outputs.logits)
        else:
            logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]

        id2label = model.config.id2label
        return {id2label[i]: prob.item() for i, prob in enumerate(probabilities)}
    except Exception as e:
        logger.error(f"Inference error (text: {text[:30]}...): {e}")
        return {}

# --- 4a-1. Bid eligibility calibration ---
def calculate_bid_possibility(binary_output: Dict[str, float]) -> float:

    logger.info(f"binary_output total: {binary_output}")

    if not binary_output:
        return 50.0

    binary_pred = max(binary_output, key=binary_output.get)
    binary_conf = binary_output[binary_pred]

    logger.info(f"prediction: {binary_pred}, confidence: {binary_conf}")

    confidence = max(binary_conf, 0.5)

    logger.info(f"final confidence: {confidence}")

    if binary_pred == 'LABEL_1':
        possibility = np.interp(confidence, [0.5, 1.0], [0.6, 1.0])
    elif binary_pred == 'LABEL_0':
        possibility = np.interp(confidence, [0.5, 1.0], [0.6, 0.0])
    else:
        possibility = 0.5

    return round(np.clip(possibility, 0, 1), 2)

# --- 4b. Multi-class model inference ---
def determine_bid_category(text: str, multi_output: Dict[str, float]) -> str:
    # Step 1: Key-Word Only Rule
    if 'technical_inspection' in text:
        return 'technical_inspection'
    if 'disaster_protection' in text:
        return 'disaster_impact_assessment'

    # Step 2: Conditional Rule
    if any(kw in text for kw in ['supervision', 'construction_management']):
        return 'electrical_comm_fire_supervision' if multi_output.get('electrical_comm_fire_supervision', 0.0) >= 0.65 else 'construction_management'

    if any(kw in text for kw in ['inspection', 'check', 'seismic_performance_assessment']):
        if multi_output.get('traffic_safety_inspection', 0.0) >= 0.65:
            return 'traffic_safety_inspection'
        if multi_output.get('water_resources_safety_inspection', 0.0) >= 0.65:
            return 'water_resources_safety_inspection'
        if multi_output.get('railway_track_safety_inspection', 0.0) >= 0.65:
            return 'railway_track_safety_inspection'
        return 'structural_safety_inspection'

    if 'impact_assessment' in text:
        if multi_output.get('power_impact_assessment', 0.0) >= 0.65:
            return 'power_impact_assessment'
        if multi_output.get('traffic_impact_assessment', 0.0) >= 0.65:
            return 'traffic_impact_assessment'
        if multi_output.get('disaster_impact_assessment', 0.0) >= 0.65:
            return 'disaster_impact_assessment'
        return 'environmental_impact_assessment'

    # Step 3: AI prediction
    if not multi_output:
        return "unclassified"
    best_label = max(multi_output, key=multi_output.get)
    best_confidence = multi_output[best_label]
    return best_label if best_confidence >= 0.6 else "unclassified"

async def process_single_notice(notice: NoticeInput) -> ClassificationResult:
    """Run all analysis for a single notice and return the final result."""

    loop = asyncio.get_running_loop()
    binary_task = loop.run_in_executor(None, run_single_prediction, models["binary_model"], models["binary_tokenizer"], notice.text)
    multi_task = loop.run_in_executor(None, run_single_prediction, models["multi_model"], models["multi_tokenizer"], notice.text)

    binary_output, multi_output = await asyncio.gather(binary_task, multi_task)

    final_possibility = calculate_bid_possibility(binary_output)
    final_category = determine_bid_category(notice.text, multi_output)

    return ClassificationResult(
        original_id=notice.id,
        possibility=final_possibility,
        category=final_category
    )

# ==============================================================================
# 5. Main API endpoint
# ==============================================================================
@app.post("/classify_batch", response_model=List[ClassificationResult])
async def classify_batch_endpoint(request: ClassificationRequest):
    """Accept a list of notices, run analysis in parallel for all notices,
    and return a list of final analysis results (bid eligibility, bid category)."""
    if not models:
        raise HTTPException(status_code=503, detail="Models have not loaded yet or loading failed.")

    # Build async task list to process each notice
    tasks = [process_single_notice(notice) for notice in request.notices]

    results = await asyncio.gather(*tasks)

    logger.info(f"Total {len(request.notices)} notices processed.")

    return results

# ==============================================================================
# 6. Run server
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
