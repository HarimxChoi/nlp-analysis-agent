import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import google.generativeai as genai
import time
import sys
import os
import json
import logging
import re

# --- 0. Logger and base config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Paths and API key config ---
# All file paths managed in this section.
DATA_DIR = r'./data'
BID_LIST_FILE = os.path.join(DATA_DIR, 'bid_list.xlsx')
KNOWLEDGE_BASE_FILE = os.path.join(DATA_DIR, 'final_clustered_label.xlsx')
OUTPUT_FILE = os.path.join(DATA_DIR, 'final_classification_with_reasoning.xlsx')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

# --- 2. Asset loading and pre-processing ---

logger.info("=" * 20 + " System init and asset loading start " + "=" * 20)

# LLM client config
gemini_client = None
try:
    if not GEMINI_API_KEY or len(GEMINI_API_KEY) < 20:
        raise ValueError("GEMINI_API_KEY is not valid")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_client = genai.GenerativeModel(GEMINI_MODEL_ID)
    logger.info(f"Gemini client configured. model: {GEMINI_MODEL_ID}")
except Exception as e:
    logger.warning(f"[WARN] Gemini client config failed: {e}. Skipping LLM-based reasoning.")

# embedding model loading
try:
    MODEL_NAME = "klue/roberta-large"
    embedding_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(MODEL_NAME)
    embedding_model.eval()  # inference mode
    logger.info(f"Embedding model loaded: {MODEL_NAME}")
except Exception as e:
    logger.error(f"[FATAL] Embedding model load failed: {e}")
    sys.exit(1)

# data loading
try:
    bid_df = pd.read_excel(BID_LIST_FILE)
    logger.info(f"Analysis target data loaded: {len(bid_df)} notices")

    knowledge_base_df = pd.read_excel(KNOWLEDGE_BASE_FILE, sheet_name='Sheet1')
    knowledge_base_df.rename(columns={'label': 'Final_Label'}, inplace=True, errors='ignore')
    logger.info("Master knowledge base loaded.")
except FileNotFoundError as e:
    logger.error(f"[FATAL] Data file load failed: {e}. Verify the path.")
    sys.exit(1)
except Exception as e:
    logger.error(f"[FATAL] Data processing exception: {e}")
    sys.exit(1)


# --- 3. Core function definitions ---

def get_sentence_embedding(text, tokenizer, model):
    """Take a sentence text as input, return the mean-pooled embedding vector."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def classify_with_llm_reasoning(title, knowledge_base, tokenizer, embedding_model, llm_client, top_n=5):
    """Use the LLM as the final judge to classify a given notice_name into the most appropriate label,
    and also generate the rationale for the judgment."""
    if not title or not isinstance(title, str):
        return 'input error', 'input notice_name is empty', None
    if not llm_client:
        return 'LLM not set', 'LLM client is not configured', None

    try:
        # Step 1: similarity-based candidate filtering
        title_embedding = get_sentence_embedding(title, tokenizer, embedding_model)
        labels = knowledge_base['Final_Label'].values
        label_embeddings = np.vstack(knowledge_base['Keyword_Embedding'].values)

        similarities = cosine_similarity([title_embedding], label_embeddings)[0]
        top_indices = similarities.argsort()[-top_n:][::-1]

        candidate_labels_info = []
        for i in top_indices:
            label_info = knowledge_base.iloc[i]
            candidate_labels_info.append(f"- Label: {label_info['Final_Label']}\n  Keywords: {label_info['Keywords_Str'][:150]}...")
        candidates_str = "\n".join(candidate_labels_info)

        # Step 2: Request final judgment from LLM (V7 prompt)
        prompt = f"""
        [CONTEXT]
        You are a Principal Strategic Consultant at a top-tier Korean civil engineering firm. Your task is to make the final, definitive classification for a new bid title based on the provided candidates.

        [BID TITLE TO CLASSIFY]
        "{title}"

        [CANDIDATE LABELS (Ranked by semantic similarity)]
        {candidates_str}

        [INSTRUCTIONS]
        1.  Carefully read the bid title.
        2.  Review the candidate labels and their associated keywords.
        3.  Choose the **single most appropriate label** from the candidates.
        4.  Briefly state the **key reason** for your choice in one short Korean sentence.
        5.  Your response **MUST BE** in the following JSON format ONLY. Do not add any other text.
            {{
              "best_label": "The single best label you chose",
              "reason": "Your short reasoning in Korean"
            }}

        [YOUR RESPONSE (JSON format ONLY)]
        """

        response = llm_client.generate_content(contents=prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
        time.sleep(1)

        response_text = response.parts[0].text.strip()
        result_json = json.loads(response_text.replace("```json", "").replace("```", "").strip())
        return result_json.get("best_label", "parse error"), result_json.get("reason", ""), candidates_str

    except Exception as e:
        logger.error(f"'{title[:30]}...' LLM classification error: {e}")
        return 'classification error', str(e), None

# --- 4. Main run block ---

def main():
    """Main function that runs the full pipeline."""
    logger.info("=" * 10 + " Knowledge base pre-processing start " + "=" * 10)
    # Concat the knowledge base's keywords into a single string, and pre-compute representative embedding per label
    keyword_cols = [col for col in knowledge_base_df.columns if 'Keyword' in col]
    knowledge_base_df['Keywords_Str'] = knowledge_base_df[keyword_cols].apply(
        lambda x: ', '.join(x.dropna().astype(str)), axis=1
    )
    knowledge_base_df['Keyword_Embedding'] = list(tqdm(
        knowledge_base_df['Keywords_Str'].apply(lambda x: get_sentence_embedding(x, embedding_tokenizer, embedding_model)),
        total=len(knowledge_base_df),
        desc="Knowledge base embedding"
    ))
    logger.info("Knowledge base embedding done.")


    logger.info("=" * 10 + " notice_name auto-classification pipeline start " + "=" * 10)
    if 'service_title' not in bid_df.columns:
        logger.error("[ERROR] 'bid_list.xlsx' missing 'service_title' column.")
        return

    tqdm.pandas(desc="Classifying notices")
    results = bid_df['service_title'].progress_apply(
        lambda title: classify_with_llm_reasoning(title, knowledge_base_df, embedding_tokenizer, embedding_model, gemini_client)
    )
    bid_df[['Predicted_Label', 'Reasoning', 'Candidates']] = pd.DataFrame(results.tolist(), index=bid_df.index)
    logger.info("Auto-classification done.")


    logger.info("=" * 10 + " Final result generation and save " + "=" * 10)
    # Map 'category' and 'managing dept' from the predicted label
    label_info_df = knowledge_base_df.set_index('Final_Label')
    bid_df['Predicted_Div'] = bid_df['Predicted_Label'].map(label_info_df.get('category'))
    bid_df['Predicted_Dept'] = bid_df['Predicted_Label'].map(label_info_df.get('managing_dept'))

    print("\n--- Auto-classification results (top 20) ---")
    print(bid_df[['service_title', 'Predicted_Label', 'Predicted_Div', 'Predicted_Dept', 'Reasoning']].head(20))

    try:
        bid_df.to_excel(OUTPUT_FILE, index=False)
        logger.info(f"All tasks done. Results saved to: {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Result file save failed: {e}")

# --- Run script ---
if __name__ == '__main__':
    main()
